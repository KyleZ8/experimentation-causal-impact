"""Balance diagnostics shared by the causal-inference cases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    """Sample-ratio-mismatch chi-square result."""

    statistic: float
    p_value: float
    observed: tuple[float, ...]
    expected: tuple[float, ...]
    has_mismatch: bool


def standardized_mean_difference(
    treated: pd.Series | np.ndarray,
    control: pd.Series | np.ndarray,
) -> float:
    """Return the standardized mean difference between treated and control values."""

    treated_values = pd.Series(treated).dropna().astype(float)
    control_values = pd.Series(control).dropna().astype(float)
    pooled_var = (treated_values.var(ddof=1) + control_values.var(ddof=1)) / 2
    if pooled_var <= 0 or np.isnan(pooled_var):
        return 0.0
    return float((treated_values.mean() - control_values.mean()) / np.sqrt(pooled_var))


def balance_table(
    frame: pd.DataFrame,
    treatment_col: str,
    covariates: Iterable[str],
) -> pd.DataFrame:
    """Compute treated/control means and absolute SMDs for covariates."""

    rows = []
    treated_mask = frame[treatment_col].astype(int).eq(1)
    for covariate in covariates:
        treated = frame.loc[treated_mask, covariate]
        control = frame.loc[~treated_mask, covariate]
        smd = standardized_mean_difference(treated, control)
        rows.append(
            {
                "covariate": covariate,
                "treated_mean": float(treated.mean()),
                "control_mean": float(control.mean()),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def srm_chisquare(
    observed: Iterable[int | float],
    expected_proportions: Iterable[int | float],
    alpha: float = 0.05,
) -> SRMResult:
    """Run an SRM chi-square test for observed assignment counts.

    The expected proportions may be probabilities, ratios, or raw allocation weights;
    they are normalized internally to the observed total.
    """

    observed_array = np.asarray(list(observed), dtype=float)
    proportions = np.asarray(list(expected_proportions), dtype=float)
    if observed_array.ndim != 1 or proportions.ndim != 1:
        raise ValueError("observed and expected_proportions must be one-dimensional")
    if len(observed_array) != len(proportions):
        raise ValueError("observed and expected_proportions must have the same length")
    if (observed_array < 0).any() or (proportions < 0).any():
        raise ValueError("observed counts and expected proportions must be non-negative")
    if observed_array.sum() <= 0 or proportions.sum() <= 0:
        raise ValueError("observed counts and expected proportions must have positive totals")

    expected = observed_array.sum() * proportions / proportions.sum()
    statistic, p_value = stats.chisquare(f_obs=observed_array, f_exp=expected)
    return SRMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        observed=tuple(float(value) for value in observed_array),
        expected=tuple(float(value) for value in expected),
        has_mismatch=bool(p_value < alpha),
    )
