"""Build case 3 artifacts from the seeded PSM data generator."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_toolkit.balance import balance_table  # noqa: E402
from causal_toolkit.psm import (  # noqa: E402
    DEFAULT_COVARIATES,
    TRUE_DIRECTED_SEARCH_EFFECT,
    estimate_psm_att,
    generate_case3_sessions,
    matched_sample,
    naive_difference,
)


def _memo_text(
    naive_effect: float,
    psm_effect: float,
    ci_low: float,
    ci_high: float,
    before_smd: float,
    after_smd: float,
    n_pairs: int,
) -> str:
    return f"""# Case 3 memo — directed-search causal impact

Question: does directed search increase session sales, or are high-intent customers simply more likely to use it?

Naive answer: sessions using directed search show ${naive_effect:.2f} more sales than other sessions.

Why that is wrong: high-intent and returning customers self-select into directed search, creating pre-treatment imbalance.

Method: estimate propensity scores from pre-session covariates, nearest-neighbor match treated sessions to comparable controls, and verify balance.

Result: the matched ATT is ${psm_effect:.2f} per treated session (95% CI ${ci_low:.2f} to ${ci_high:.2f}); the planted effect is ${TRUE_DIRECTED_SEARCH_EFFECT:.2f}.

Recommendation: credit directed search with the matched effect, not the naive gap; the mean absolute SMD falls from {before_smd:.3f} to {after_smd:.3f} across {n_pairs:,} matched pairs.

Limitations: PSM only adjusts observed covariates, depends on overlap, and should be paired with product knowledge about unobserved search intent.
"""


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    sessions = generate_case3_sessions()
    naive = naive_difference(sessions)
    result, pairs, scored = estimate_psm_att(sessions)
    sessions.to_csv(data_dir / "case3_sessions.csv", index=False)
    pairs.to_csv(data_dir / "case3_psm_matches.csv", index=False)

    matched = matched_sample(scored, pairs)
    before = balance_table(scored, "directed_search_usage", DEFAULT_COVARIATES)
    after = balance_table(matched, "directed_search_usage", DEFAULT_COVARIATES)
    balance = pd.concat(
        [
            before.assign(sample="before_matching"),
            after.assign(sample="after_matching"),
        ],
        ignore_index=True,
    )
    balance.to_csv(data_dir / "case3_balance.csv", index=False)

    plot_frame = balance.pivot(index="covariate", columns="sample", values="abs_smd").sort_values(
        "before_matching"
    )
    ax = plot_frame.plot(kind="barh", figsize=(8, 4))
    ax.axvline(0.10, color="black", linestyle="--", linewidth=1)
    ax.set_title("Case 3: covariate balance before and after PSM")
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_ylabel("")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(figure_dir / "case3_balance.png", dpi=160)
    plt.close()

    memo = _memo_text(
        naive_effect=naive,
        psm_effect=result.att,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        before_smd=result.mean_abs_smd_before,
        after_smd=result.mean_abs_smd_after,
        n_pairs=result.n_pairs,
    )
    (PROJECT_ROOT / "cases" / "03_psm" / "MEMO.md").write_text(memo, encoding="utf-8")
    print(
        "case3 "
        f"naive_effect={naive:.4f} "
        f"psm_att={result.att:.4f} "
        f"pairs={result.n_pairs} "
        f"mean_abs_smd={result.mean_abs_smd_before:.4f}->{result.mean_abs_smd_after:.4f}"
    )


if __name__ == "__main__":
    main()
