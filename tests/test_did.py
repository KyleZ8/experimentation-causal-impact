from __future__ import annotations

import pytest

from causal_toolkit.did import (
    TRUE_INCREMENTAL_CLICKS,
    check_parallel_trends,
    did_incremental_roi,
    estimate_did,
    filter_parallel_comparison,
    generate_case2_panel,
    naive_paid_search_roi,
    simple_did,
)


def test_case2_did_recovers_the_planted_incremental_click_effect() -> None:
    panel = filter_parallel_comparison(generate_case2_panel())

    result = estimate_did(panel)

    assert result.estimate == pytest.approx(TRUE_INCREMENTAL_CLICKS, abs=0.01)
    assert result.ci_low <= TRUE_INCREMENTAL_CLICKS <= result.ci_high


def test_case2_naive_roi_is_inflated_by_paid_organic_substitution() -> None:
    panel = filter_parallel_comparison(generate_case2_panel())
    did = estimate_did(panel)

    assert naive_paid_search_roi(panel) == pytest.approx(3.2)
    assert did_incremental_roi(panel, did.estimate) == pytest.approx(0.8, abs=0.001)


def test_case2_parallel_trend_screen_rejects_the_broad_control_pool() -> None:
    panel = generate_case2_panel()

    broad = check_parallel_trends(panel)
    matched = check_parallel_trends(filter_parallel_comparison(panel))

    assert broad.passes is False
    assert matched.passes is True


def test_case2_simple_did_matches_the_fixed_effect_estimate_on_parallel_controls() -> None:
    panel = filter_parallel_comparison(generate_case2_panel())

    assert simple_did(panel) == pytest.approx(estimate_did(panel).estimate)
