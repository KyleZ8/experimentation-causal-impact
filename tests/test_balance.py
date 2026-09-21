from __future__ import annotations

import pytest

from causal_toolkit.balance import srm_chisquare, standardized_mean_difference


def test_srm_passes_for_expected_traffic_split() -> None:
    result = srm_chisquare([5000, 5000], [0.5, 0.5])

    assert result.has_mismatch is False
    assert result.p_value == pytest.approx(1.0)


def test_srm_flags_sample_ratio_mismatch() -> None:
    result = srm_chisquare([5600, 4400], [0.5, 0.5])

    assert result.has_mismatch is True
    assert result.p_value < 0.001


def test_standardized_mean_difference_is_zero_when_both_groups_are_constant() -> None:
    assert standardized_mean_difference([1, 1, 1], [1, 1, 1]) == 0.0
