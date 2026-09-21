"""Propensity-score matching helpers for the directed-search case."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_toolkit.balance import balance_table

TRUE_DIRECTED_SEARCH_EFFECT = 8.0
DEFAULT_COVARIATES = [
    "prior_sessions",
    "high_intent",
    "returning_customer",
    "mobile_session",
    "prior_customer_value",
]


@dataclass(frozen=True)
class PSMResult:
    """ATT estimate from nearest-neighbor propensity-score matching."""

    att: float
    std_error: float
    ci_low: float
    ci_high: float
    n_pairs: int
    mean_abs_smd_before: float
    mean_abs_smd_after: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_case3_sessions(seed: int = 20261003, n_sessions: int = 6_000) -> pd.DataFrame:
    """Generate seeded session-level data with self-selection into directed search."""

    rng = np.random.default_rng(seed)
    prior_sessions = rng.poisson(3.0, n_sessions)
    high_intent = rng.binomial(1, _sigmoid(-0.8 + 0.24 * prior_sessions))
    returning_customer = rng.binomial(1, _sigmoid(-0.4 + 0.18 * prior_sessions))
    mobile_session = rng.binomial(1, 0.54, n_sessions)
    prior_customer_value = np.clip(
        rng.normal(90 + 8 * prior_sessions + 18 * returning_customer, 22, n_sessions),
        5,
        None,
    )

    treatment_score = (
        -2.35
        + 0.30 * prior_sessions
        + 0.95 * high_intent
        + 0.55 * returning_customer
        - 0.25 * mobile_session
        + 0.006 * (prior_customer_value - 90)
    )
    propensity = _sigmoid(treatment_score)
    directed_search = rng.binomial(1, propensity)

    baseline_sales = (
        18
        + 2.15 * prior_sessions
        + 9.5 * high_intent
        + 5.5 * returning_customer
        - 1.8 * mobile_session
        + 0.055 * prior_customer_value
    )
    sales = baseline_sales + TRUE_DIRECTED_SEARCH_EFFECT * directed_search + rng.normal(
        0, 1.7, n_sessions
    )

    return pd.DataFrame(
        {
            "session_id": [f"S{i:05d}" for i in range(1, n_sessions + 1)],
            "directed_search_usage": directed_search.astype(int),
            "sales": np.round(np.clip(sales, 0, None), 4),
            "prior_sessions": prior_sessions.astype(int),
            "high_intent": high_intent.astype(int),
            "returning_customer": returning_customer.astype(int),
            "mobile_session": mobile_session.astype(int),
            "prior_customer_value": np.round(prior_customer_value, 4),
            "true_effect": TRUE_DIRECTED_SEARCH_EFFECT,
            "true_propensity": np.round(propensity, 6),
        }
    )


def naive_difference(
    frame: pd.DataFrame,
    outcome_col: str = "sales",
    treatment_col: str = "directed_search_usage",
) -> float:
    """Unadjusted treated-minus-control outcome difference."""

    treated = frame[frame[treatment_col].eq(1)][outcome_col]
    control = frame[frame[treatment_col].eq(0)][outcome_col]
    return float(treated.mean() - control.mean())


def estimate_propensity_scores(
    frame: pd.DataFrame,
    treatment_col: str,
    covariates: Iterable[str],
) -> pd.Series:
    """Fit a logistic propensity model."""

    x = sm.add_constant(frame[list(covariates)], has_constant="add")
    y = frame[treatment_col].astype(int)
    model = sm.Logit(y, x).fit(disp=False)
    return pd.Series(model.predict(x), index=frame.index, name="propensity_score").clip(0.001, 0.999)


def nearest_neighbor_pairs(
    frame: pd.DataFrame,
    treatment_col: str = "directed_search_usage",
    score_col: str = "propensity_score",
    outcome_col: str = "sales",
    caliper: float = 0.025,
) -> pd.DataFrame:
    """Create one nearest control match for each treated row within a propensity caliper."""

    treated = frame[frame[treatment_col].eq(1)].copy()
    controls = frame[frame[treatment_col].eq(0)].copy().sort_values(score_col)
    control_scores = controls[score_col].to_numpy()
    control_indices = controls.index.to_numpy()
    pair_rows = []

    for pair_id, (treated_index, treated_row) in enumerate(treated.iterrows(), start=1):
        score = float(treated_row[score_col])
        insert_at = int(np.searchsorted(control_scores, score))
        candidates = []
        if insert_at < len(control_scores):
            candidates.append(insert_at)
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if not candidates:
            continue
        best_position = min(candidates, key=lambda pos: abs(control_scores[pos] - score))
        distance = abs(control_scores[best_position] - score)
        if distance > caliper:
            continue
        control_index = control_indices[best_position]
        control_row = frame.loc[control_index]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "treated_index": treated_index,
                "control_index": control_index,
                "treated_score": score,
                "control_score": float(control_row[score_col]),
                "score_distance": float(distance),
                "treated_outcome": float(treated_row[outcome_col]),
                "control_outcome": float(control_row[outcome_col]),
                "difference": float(treated_row[outcome_col] - control_row[outcome_col]),
            }
        )

    return pd.DataFrame(pair_rows)


def matched_sample(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    treatment_col: str = "directed_search_usage",
) -> pd.DataFrame:
    """Return a stacked treated/control sample from matched pair indices."""

    treated = frame.loc[pairs["treated_index"]].copy()
    control = frame.loc[pairs["control_index"]].copy()
    treated["_matched_pair_id"] = pairs["pair_id"].to_numpy()
    control["_matched_pair_id"] = pairs["pair_id"].to_numpy()
    treated[treatment_col] = 1
    control[treatment_col] = 0
    return pd.concat([treated, control], ignore_index=True)


def estimate_psm_att(
    frame: pd.DataFrame,
    covariates: Iterable[str] = DEFAULT_COVARIATES,
    treatment_col: str = "directed_search_usage",
    outcome_col: str = "sales",
    caliper: float = 0.025,
) -> tuple[PSMResult, pd.DataFrame, pd.DataFrame]:
    """Estimate ATT with nearest-neighbor PSM and return diagnostics."""

    scored = frame.copy()
    scored["propensity_score"] = estimate_propensity_scores(scored, treatment_col, covariates)
    pairs = nearest_neighbor_pairs(scored, treatment_col, "propensity_score", outcome_col, caliper)
    if pairs.empty:
        raise ValueError("No matches found inside the propensity caliper")

    differences = pairs["difference"]
    att = float(differences.mean())
    std_error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    ci_low = att - 1.96 * std_error
    ci_high = att + 1.96 * std_error

    before = balance_table(scored, treatment_col, covariates)
    after = balance_table(matched_sample(scored, pairs, treatment_col), treatment_col, covariates)
    result = PSMResult(
        att=att,
        std_error=std_error,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_pairs=int(len(pairs)),
        mean_abs_smd_before=float(before["abs_smd"].mean()),
        mean_abs_smd_after=float(after["abs_smd"].mean()),
    )
    return result, pairs, scored
