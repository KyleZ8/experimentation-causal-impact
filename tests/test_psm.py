from __future__ import annotations

import pytest

from causal_toolkit.psm import (
    DEFAULT_COVARIATES,
    TRUE_DIRECTED_SEARCH_EFFECT,
    estimate_psm_att,
    generate_case3_sessions,
    naive_difference,
)


def test_case3_naive_gap_overstates_the_planted_directed_search_effect() -> None:
    sessions = generate_case3_sessions()

    assert naive_difference(sessions) > TRUE_DIRECTED_SEARCH_EFFECT + 4


def test_case3_psm_recovers_the_planted_directed_search_effect() -> None:
    sessions = generate_case3_sessions()

    result, pairs, scored = estimate_psm_att(sessions)

    assert result.att == pytest.approx(TRUE_DIRECTED_SEARCH_EFFECT, abs=0.35)
    assert result.ci_low <= TRUE_DIRECTED_SEARCH_EFFECT <= result.ci_high
    assert len(pairs) == result.n_pairs
    assert set(DEFAULT_COVARIATES).issubset(scored.columns)


def test_case3_psm_improves_covariate_balance() -> None:
    sessions = generate_case3_sessions()

    result, _, _ = estimate_psm_att(sessions)

    assert result.mean_abs_smd_after < 0.10
    assert result.mean_abs_smd_after < result.mean_abs_smd_before / 4
