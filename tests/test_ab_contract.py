from __future__ import annotations

import pytest

ab = pytest.importorskip("causal_toolkit.ab", reason="case 1 is being built in parallel")


def test_case1_ab_module_exposes_an_srm_check() -> None:
    assert "srm_check" in dir(ab)


def test_case1_ab_module_exposes_a_known_answer_readout() -> None:
    assert {"naive_effect", "adjusted_effect"}.issubset(dir(ab))
