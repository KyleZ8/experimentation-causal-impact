"""Difference-in-differences helpers for the sponsored-search case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TRUE_INCREMENTAL_CLICKS = 45.0
PAID_CLICKS_PER_CELL = 180.0
SPEND_PER_CELL = 112.5
REVENUE_PER_CLICK = 2.0


@dataclass(frozen=True)
class DIDResult:
    """Estimated DiD treatment effect."""

    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_rows: int


@dataclass(frozen=True)
class ParallelTrendResult:
    """Pre-period slope comparison."""

    treated_slope: float
    control_slope: float
    slope_difference: float
    passes: bool


def generate_case2_panel(
    seed: int = 20261002,
    n_treated: int = 40,
    n_matched_controls: int = 20,
    n_nonparallel_controls: int = 20,
) -> pd.DataFrame:
    """Generate seeded synthetic market-week data with a planted paid-search effect."""

    rng = np.random.default_rng(seed)
    weeks = np.arange(1, 21)
    post_start = 13
    rows: list[dict[str, float | int | str]] = []
    groups = (
        [("treated", 1, n_treated)]
        + [("matched_control", 0, n_matched_controls)]
        + [("nonparallel_control", 0, n_nonparallel_controls)]
    )

    market_number = 1
    for group, treated_market, count in groups:
        for _ in range(count):
            market_id = f"M{market_number:03d}"
            market_number += 1
            base = rng.normal(980, 85)
            market_shift = rng.normal(0, 22)
            if group == "nonparallel_control":
                weekly_slope = 3.0
            else:
                weekly_slope = 8.0

            for week in weeks:
                post = int(week >= post_start)
                seasonal = 18 * np.sin(week / 2.5)
                baseline_clicks = base + market_shift + weekly_slope * week + seasonal
                paid_clicks = 0.0
                spend = 0.0
                true_incremental_clicks = 0.0
                substitution_clicks = 0.0
                if treated_market and post:
                    paid_clicks = PAID_CLICKS_PER_CELL
                    spend = SPEND_PER_CELL
                    true_incremental_clicks = TRUE_INCREMENTAL_CLICKS
                    substitution_clicks = PAID_CLICKS_PER_CELL - TRUE_INCREMENTAL_CLICKS

                total_clicks = baseline_clicks + true_incremental_clicks
                organic_clicks = total_clicks - paid_clicks
                rows.append(
                    {
                        "market_id": market_id,
                        "week": int(week),
                        "group": group,
                        "treated_market": int(treated_market),
                        "post": int(post),
                        "eligible_control": int(group != "nonparallel_control"),
                        "organic_clicks": round(float(organic_clicks), 4),
                        "paid_clicks": round(float(paid_clicks), 4),
                        "total_clicks": round(float(total_clicks), 4),
                        "spend": round(float(spend), 2),
                        "revenue": round(float(total_clicks * REVENUE_PER_CLICK), 4),
                        "paid_revenue": round(float(paid_clicks * REVENUE_PER_CLICK), 4),
                        "true_incremental_clicks": round(float(true_incremental_clicks), 4),
                        "substitution_clicks": round(float(substitution_clicks), 4),
                    }
                )
    return pd.DataFrame(rows)


def filter_parallel_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep treated markets and the comparator markets designed to pass pre-trends."""

    return frame[frame["eligible_control"].eq(1)].copy()


def estimate_did(
    frame: pd.DataFrame,
    outcome_col: str = "total_clicks",
    robust_covariance: str = "HC1",
) -> DIDResult:
    """Estimate a two-way fixed-effect DiD model and return the interaction effect."""

    model = smf.ols(
        f"{outcome_col} ~ treated_market * post + C(market_id) + C(week)",
        data=frame,
    ).fit(cov_type=robust_covariance)
    term = "treated_market:post"
    ci_low, ci_high = model.conf_int().loc[term].tolist()
    return DIDResult(
        estimate=float(model.params[term]),
        std_error=float(model.bse[term]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(model.pvalues[term]),
        n_rows=int(model.nobs),
    )


def simple_did(frame: pd.DataFrame, outcome_col: str = "total_clicks") -> float:
    """Compute the 2x2 DiD contrast from group means."""

    means = frame.groupby(["treated_market", "post"])[outcome_col].mean()
    return float((means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0]))


def check_parallel_trends(
    frame: pd.DataFrame,
    outcome_col: str = "total_clicks",
    max_abs_slope_difference: float = 0.75,
) -> ParallelTrendResult:
    """Compare treated and control pre-period slopes."""

    pre = frame[frame["post"].eq(0)]
    slopes = {}
    for treated_value, group in pre.groupby("treated_market"):
        centered_week = group["week"] - group["week"].mean()
        centered_outcome = group[outcome_col] - group[outcome_col].mean()
        slope = (centered_week * centered_outcome).sum() / (centered_week**2).sum()
        slopes[int(treated_value)] = float(slope)
    difference = slopes[1] - slopes[0]
    return ParallelTrendResult(
        treated_slope=slopes[1],
        control_slope=slopes[0],
        slope_difference=float(difference),
        passes=bool(abs(difference) <= max_abs_slope_difference),
    )


def naive_paid_search_roi(frame: pd.DataFrame) -> float:
    """ROI if every paid click is treated as incremental."""

    treated_post = frame[frame["treated_market"].eq(1) & frame["post"].eq(1)]
    return float(treated_post["paid_revenue"].sum() / treated_post["spend"].sum())


def did_incremental_roi(frame: pd.DataFrame, effect_clicks: float) -> float:
    """Translate an incremental click estimate into revenue per dollar of spend."""

    treated_post = frame[frame["treated_market"].eq(1) & frame["post"].eq(1)]
    spend_per_cell = treated_post["spend"].mean()
    return float(effect_clicks * REVENUE_PER_CLICK / spend_per_cell)
