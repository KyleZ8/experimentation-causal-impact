"""Build case 2 artifacts from the seeded DiD data generator."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_toolkit.did import (  # noqa: E402
    TRUE_INCREMENTAL_CLICKS,
    check_parallel_trends,
    did_incremental_roi,
    estimate_did,
    filter_parallel_comparison,
    generate_case2_panel,
    naive_paid_search_roi,
)


def _memo_text(
    naive_roi: float,
    broad_pretrend_passes: bool,
    matched_pretrend_passes: bool,
    did_effect: float,
    ci_low: float,
    ci_high: float,
    did_roi: float,
) -> str:
    return f"""# Case 2 memo — paid-search incrementality

Question: should the sponsored-search program be scaled based on the observed paid-click ROI?

Naive answer: paid clicks appear to return {naive_roi:.1%} revenue per dollar of spend because every paid click is counted as incremental.

Why that is wrong: the campaign cannibalizes organic clicks, and the broad comparison pool fails the pre-period parallel-trends screen (`passes={broad_pretrend_passes}`).

Method: use a difference-in-differences model on total clicks, restricted to the comparator markets that pass the same pre-trends check (`passes={matched_pretrend_passes}`).

Result: the estimated incremental lift is {did_effect:.1f} clicks per treated market-week (95% CI {ci_low:.1f} to {ci_high:.1f}); the planted effect is {TRUE_INCREMENTAL_CLICKS:.1f}.

Recommendation: use the DiD readout, not the paid-click topline; the corrected ROI is {did_roi:.1%}, so scaling should depend on whether that clears the margin hurdle.

Limitations: this synthetic case assumes stable market composition, no spillovers, and a validated comparable-control screen.
"""


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    panel = generate_case2_panel()
    panel.to_csv(data_dir / "case2_did_panel.csv", index=False)

    broad_trend = check_parallel_trends(panel)
    matched_panel = filter_parallel_comparison(panel)
    matched_trend = check_parallel_trends(matched_panel)
    did = estimate_did(matched_panel)
    naive_roi = naive_paid_search_roi(panel)
    correct_roi = did_incremental_roi(matched_panel, did.estimate)

    trend = (
        panel.groupby(["week", "group"], as_index=False)["total_clicks"]
        .mean()
        .pivot(index="week", columns="group", values="total_clicks")
    )
    ax = trend.plot(figsize=(8, 4), linewidth=2)
    ax.axvline(12.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Case 2: paid-search DiD pre-trends and post-period lift")
    ax.set_xlabel("Week")
    ax.set_ylabel("Average total clicks")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(figure_dir / "case2_parallel_trends.png", dpi=160)
    plt.close()

    memo = _memo_text(
        naive_roi=naive_roi,
        broad_pretrend_passes=broad_trend.passes,
        matched_pretrend_passes=matched_trend.passes,
        did_effect=did.estimate,
        ci_low=did.ci_low,
        ci_high=did.ci_high,
        did_roi=correct_roi,
    )
    (PROJECT_ROOT / "cases" / "02_did" / "MEMO.md").write_text(memo, encoding="utf-8")
    print(
        "case2 "
        f"naive_roi={naive_roi:.4f} "
        f"did_clicks={did.estimate:.4f} "
        f"did_roi={correct_roi:.4f} "
        f"broad_pretrend_passes={broad_trend.passes}"
    )


if __name__ == "__main__":
    main()
