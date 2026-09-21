from __future__ import annotations

import pytest

ab = pytest.importorskip("causal_toolkit.ab", reason="case 1 is being built in parallel")


def test_case1_ab_module_exposes_an_srm_check() -> None:
    assert any(name in dir(ab) for name in ("srm_chisquare", "check_srm", "sample_ratio_mismatch"))


def test_case1_ab_module_exposes_a_known_answer_readout() -> None:
    assert any(name in dir(ab) for name in ("estimate_ab_effect", "readout_ab_test", "analyze_ab_test"))
