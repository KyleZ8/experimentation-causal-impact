"""Reusable A/B test readout functions: SRM, power, effect + CI, guardrails.

Case 1 mirrors the schema and rough effect size of the C22/HW1 course case
(an online platform's posting-activity A/B test) with seeded synthetic data
and a planted true effect, since the source data has no redistribution
license. See ../../.ai/PROJECT_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "case1"

# Ground truth planted in the synthetic generator, used by the notebook to
# state "true effect" alongside what the estimators recover.
TRUE_PRIMARY_LIFT_PP = 6.0  # percentage points, posting rate
TRUE_GUARDRAIL_LENGTH_PCT = -10.0  # percent change in avg post length
TRUE_GUARDRAIL_FLAG_PP = 1.2  # percentage points, content-flag rate


def generate_case1_data(n: int = 20_000, seed: int = 20260101) -> pd.DataFrame:
    """Simulate a two-arm posting-activity A/B test.

    Randomization is a fair coin flip per user (not forced to an exact
    50/50 split), so the SRM check below is a real check, not a formality.
    The treatment lifts posting propensity (~6pp, matching the source
    case's regression coefficient) but degrades two guardrail metrics:
    average post length and the content-moderation flag rate, so a
    topline-only readout would miss a real quality tradeoff.
    """
    rng = np.random.default_rng(seed)

    treated = rng.binomial(1, 0.5, n)
    tenure_days = rng.lognormal(mean=6.0, sigma=0.8, size=n).clip(1, 3000)
    premium_user = rng.binomial(1, 0.15, n)
    num_post_before = rng.poisson(lam=0.01 * tenure_days).clip(0, 200)

    # Primary outcome: posted (binary), logistic model calibrated so the
    # control arm posts ~35% of the time and the treatment effect is ~6pp.
    logit_p = (
        -1.33
        + 0.26 * treated
        + 0.10 * np.log1p(tenure_days)
        + 0.25 * premium_user
        + 0.05 * np.log1p(num_post_before)
    )
    p_post = 1 / (1 + np.exp(-logit_p))
    posted = rng.binomial(1, p_post)

    # Guardrail 1: post length (words), only defined for posters. Treatment
    # pushes people to post more often but with shorter, lower-effort posts.
    base_length = rng.normal(140, 35, n)
    length_effect = -14.0 * treated
    post_length = (base_length + length_effect).clip(5, None)

    # Guardrail 2: content-moderation flag (binary), only defined for
    # posters. Treatment slightly raises the flag rate.
    flag_logit = -3.4 + 0.45 * treated - 0.01 * (post_length - 140) / 35
    flagged = rng.binomial(1, 1 / (1 + np.exp(-flag_logit)))

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "treated": treated,
            "tenure_days": tenure_days.round(1),
            "premium_user": premium_user,
            "num_post_before": num_post_before,
            "posted": posted,
            "post_length": np.where(posted == 1, post_length.round(1), np.nan),
            "flagged": np.where(posted == 1, flagged, np.nan),
        }
    )
    return df


def save_case1_data(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case1_users.csv"
    df.to_csv(out_path, index=False)
    return out_path


@dataclass
class SrmResult:
    n_control: int
    n_treated: int
    chi2: float
    p_value: float
    balanced: bool


def srm_check(df: pd.DataFrame, expected_ratio: float = 0.5, alpha: float = 0.001) -> SrmResult:
    """Sample ratio mismatch check via chi-square goodness of fit.

    Uses the conventional SRM alpha of 0.001 (not 0.05): assignment
    imbalance needs strong evidence before it's treated as a bug, since
    routine random variation will cross p<0.05 fairly often.
    """
    n_treated = int(df["treated"].sum())
    n_control = int(len(df) - n_treated)
    n_total = n_treated + n_control
    expected = [n_total * (1 - expected_ratio), n_total * expected_ratio]
    chi2, p_value = stats.chisquare([n_control, n_treated], f_exp=expected)
    return SrmResult(n_control, n_treated, float(chi2), float(p_value), p_value >= alpha)


@dataclass
class PowerResult:
    baseline_rate: float
    mde_pp: float
    n_per_arm_actual: int
    n_per_arm_required: int
    achieved_power: float
    adequately_powered: bool


def power_check(
    df: pd.DataFrame,
    outcome: str = "posted",
    mde_pp: float = TRUE_PRIMARY_LIFT_PP,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> PowerResult:
    """Required vs. actual sample size for detecting `mde_pp` on `outcome`."""
    control = df.loc[df["treated"] == 0, outcome]
    baseline_rate = float(control.mean())
    effect_size = proportion_effectsize(baseline_rate + mde_pp / 100, baseline_rate)
    analysis = NormalIndPower()
    n_required = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=target_power, ratio=1.0)
    n_actual = int((df["treated"] == 0).sum())
    achieved_power = analysis.power(effect_size=effect_size, nobs1=n_actual, alpha=alpha, ratio=1.0)
    return PowerResult(
        baseline_rate=baseline_rate,
        mde_pp=mde_pp,
        n_per_arm_actual=n_actual,
        n_per_arm_required=int(np.ceil(n_required)),
        achieved_power=float(achieved_power),
        adequately_powered=n_actual >= n_required,
    )


@dataclass
class EffectResult:
    label: str
    control_rate: float
    treated_rate: float
    diff_pp: float
    ci_low_pp: float
    ci_high_pp: float
    p_value: float


def naive_effect(df: pd.DataFrame, outcome: str = "posted") -> EffectResult:
    """Unadjusted two-proportion comparison: the 'just look at the topline' read."""
    control = df.loc[df["treated"] == 0, outcome]
    treated = df.loc[df["treated"] == 1, outcome]
    rate_c, rate_t = control.mean(), treated.mean()
    diff = rate_t - rate_c

    ci_c = proportion_confint(control.sum(), len(control), method="wilson")
    ci_t = proportion_confint(treated.sum(), len(treated), method="wilson")
    se = np.sqrt(np.diff(ci_c)[0] ** 2 + np.diff(ci_t)[0] ** 2) / (2 * 1.96)
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

    pooled = (control.sum() + treated.sum()) / (len(control) + len(treated))
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / len(control) + 1 / len(treated)))
    z = diff / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return EffectResult("naive (unadjusted)", rate_c, rate_t, diff * 100, ci_low * 100, ci_high * 100, p_value)


def adjusted_effect(
    df: pd.DataFrame,
    outcome: str = "posted",
    covariates: tuple[str, ...] = ("tenure_days", "premium_user", "num_post_before"),
) -> EffectResult:
    """Covariate-adjusted average marginal effect from a logistic regression."""
    X = df[["treated", *covariates]].copy()
    X["tenure_days"] = np.log1p(X["tenure_days"])
    X["num_post_before"] = np.log1p(X["num_post_before"])
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.Logit(y, X).fit(disp=0)
    ame = model.get_margeff(at="overall")
    idx = list(X.columns[1:]).index("treated")
    diff_pp = ame.margeff[idx] * 100
    ci = ame.conf_int()[idx]
    p_value = ame.pvalues[idx]

    rate_c = float(df.loc[df["treated"] == 0, outcome].mean())
    rate_t = float(df.loc[df["treated"] == 1, outcome].mean())
    return EffectResult(
        "covariate-adjusted (logistic AME)", rate_c, rate_t, diff_pp, ci[0] * 100, ci[1] * 100, float(p_value)
    )


@dataclass
class GuardrailResult:
    metric: str
    kind: str
    control_value: float
    treated_value: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    regressed: bool


def guardrail_check_continuous(df: pd.DataFrame, metric: str, worse_if: str = "lower") -> GuardrailResult:
    sub = df.dropna(subset=[metric])
    control = sub.loc[sub["treated"] == 0, metric]
    treated = sub.loc[sub["treated"] == 1, metric]
    diff = treated.mean() - control.mean()
    t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)
    se = np.sqrt(control.var(ddof=1) / len(control) + treated.var(ddof=1) / len(treated))
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    regressed = p_value < 0.05 and ((diff < 0) if worse_if == "lower" else (diff > 0))
    return GuardrailResult(metric, "continuous", control.mean(), treated.mean(), diff, ci_low, ci_high, p_value, regressed)


def guardrail_check_binary(df: pd.DataFrame, metric: str, worse_if: str = "higher") -> GuardrailResult:
    sub = df.dropna(subset=[metric])
    control = sub.loc[sub["treated"] == 0, metric]
    treated = sub.loc[sub["treated"] == 1, metric]
    diff = treated.mean() - control.mean()
    ci_c = proportion_confint(control.sum(), len(control), method="wilson")
    ci_t = proportion_confint(treated.sum(), len(treated), method="wilson")
    se = np.sqrt(np.diff(ci_c)[0] ** 2 + np.diff(ci_t)[0] ** 2) / (2 * 1.96)
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    pooled = (control.sum() + treated.sum()) / (len(control) + len(treated))
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / len(control) + 1 / len(treated)))
    p_value = 2 * (1 - stats.norm.cdf(abs(diff / se_pooled)))
    regressed = p_value < 0.05 and ((diff > 0) if worse_if == "higher" else (diff < 0))
    return GuardrailResult(metric, "binary", control.mean(), treated.mean(), diff, ci_low, ci_high, p_value, regressed)


if __name__ == "__main__":
    data = generate_case1_data()
    path = save_case1_data(data)
    print(f"wrote {len(data):,} rows to {path}")
