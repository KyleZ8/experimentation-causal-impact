from __future__ import annotations

import pytest

from causal_toolkit.ab import (
    TRUE_PRIMARY_LIFT_PP,
    adjusted_effect,
    generate_case1_data,
    guardrail_check_binary,
    guardrail_check_continuous,
    naive_effect,
    power_check,
    srm_check,
)


def test_case1_srm_accepts_the_seeded_randomization_split() -> None:
    data = generate_case1_data()

    result = srm_check(data)

    assert result.balanced
    assert result.n_control == 10_082
    assert result.n_treated == 9_918
    assert result.p_value == pytest.approx(0.246, abs=0.001)


def test_case1_ab_readout_recovers_the_planted_posting_lift() -> None:
    data = generate_case1_data()

    naive = naive_effect(data)
    adjusted = adjusted_effect(data)

    assert naive.diff_pp == pytest.approx(TRUE_PRIMARY_LIFT_PP, abs=0.05)
    assert adjusted.diff_pp == pytest.approx(TRUE_PRIMARY_LIFT_PP, abs=0.05)
    assert naive.ci_low_pp <= TRUE_PRIMARY_LIFT_PP <= naive.ci_high_pp
    assert adjusted.ci_low_pp <= TRUE_PRIMARY_LIFT_PP <= adjusted.ci_high_pp


def test_case1_power_and_guardrails_change_the_ship_decision() -> None:
    data = generate_case1_data()

    power = power_check(data)
    post_length = guardrail_check_continuous(data, "post_length", worse_if="lower")
    flag_rate = guardrail_check_binary(data, "flagged", worse_if="higher")

    assert power.adequately_powered
    assert power.n_per_arm_required == 1_030
    assert post_length.regressed
    assert post_length.diff == pytest.approx(-14.36, abs=0.05)
    assert flag_rate.regressed
    assert flag_rate.diff * 100 == pytest.approx(1.60, abs=0.05)
