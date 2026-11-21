"""ATL_STAR_F and SL1G_F results: values per state and synthesized strategies, worked out by hand.

On matching pennies (sl1g_f_2agents_3states_pennies.txt) equal actions at s0 lead
to s1 (p = 0.8), different ones to s2 (p = 0.3), and both loop. With agent 1
choosing first, agent 2 mismatches and `X p` is worth 0.3 at s0; with agent 2
first, agent 1 matches and it is worth 0.8. Needs Spot.
"""

import json
from fractions import Fraction

import pytest

spot = pytest.importorskip("spot")

from model_checker.algorithms.explicit.ATL_STAR_F import ATL_STAR_F
from model_checker.algorithms.explicit.SL1G_F import SL1G_F

pytestmark = pytest.mark.atl_star

_PENNIES = "fuzzyCGS/SL1G_F/sl1g_f_2agents_3states_pennies.txt"
_FUZZY = "fuzzyCGS/ATLF/atlf_2agents_4states_fuzzy.txt"

_CHECKING_CASES = [
    # same values as ATLF's <1>F p: agent 2 keeps the play in s1
    (
        ATL_STAR_F.model_checking,
        "<<1>> F p",
        _FUZZY,
        {"s0": "1/5", "s1": "1/5", "s2": "7/10", "s3": "2/5"},
    ),
    (
        SL1G_F.model_checking,
        "E x A y: (x, 1)(y, 2) X p",
        _PENNIES,
        {"s0": "3/10", "s1": "4/5", "s2": "3/10"},
    ),
    (
        SL1G_F.model_checking,
        "A y E x: (x, 1)(y, 2) X p",
        _PENNIES,
        {"s0": "4/5", "s1": "4/5", "s2": "3/10"},
    ),
]


@pytest.mark.integration
@pytest.mark.model_checking
@pytest.mark.parametrize(("check", "formula", "model", "exact"), _CHECKING_CASES)
def test_checking_gives_float_and_exact_values_per_state(
    test_data_dir, check, formula, model, exact
):
    result = check(formula, str(test_data_dir / model))
    assert "error" not in result, result
    assert result["exact_values"] == exact
    assert result["state_values"] == {
        state: float(Fraction(v)) for state, v in exact.items()
    }
    assert (
        result["initial_state"] == f"Initial state s0: {result['state_values']['s0']}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("formula", "value", "s0_value"),
    [
        ("E x A y: (x, 1)(y, 2) X p", "3/10", 0.3),
        ("A y E x: (x, 1)(y, 2) X p", "4/5", 0.8),
    ],
)
def test_synthesis_carries_value_strategy_and_values_per_state(
    test_data_dir, formula, value, s0_value
):
    result = SL1G_F.synthesize_strategy(formula, str(test_data_dir / _PENNIES))
    assert "error" not in result, result
    json.dumps(result)
    assert result["satisfied"] is None
    assert result["strategy"]["value"] == value
    assert result["initial_state"] == f"State s0: {s0_value}"
    assert result["state_values"]["s0"] == s0_value
    assert result["exact_values"]["s0"] == value
    assert set(result["state_values"]) == {"s0", "s1", "s2"}


@pytest.mark.integration
def test_existential_agent_after_a_universal_one_reacts_to_its_action(test_data_dir):
    """In A y E x agent 1 matches agent 2 at s0: its move depends on `given`."""
    result = SL1G_F.synthesize_strategy(
        "A y E x: (x, 1)(y, 2) X p", str(test_data_dir / _PENNIES)
    )
    at_s0 = [m for m in result["strategy"]["moves"] if m["state"] == "s0"]
    assert {(tuple(map(tuple, m["given"])), m["action"]) for m in at_s0} == {
        (((2, "H"),), "H"),
        (((2, "T"),), "T"),
    }


@pytest.mark.integration
def test_atl_star_f_synthesis_at_another_state(test_data_dir):
    """At s2 p is 0.7 forever, so <<1>> F p is worth 0.7 there."""
    result = ATL_STAR_F.synthesize_strategy(
        "<<1>> F p", str(test_data_dir / _FUZZY), state="s2"
    )
    assert "error" not in result, result
    assert result["strategy"]["value"] == "7/10"
    assert result["strategy"]["start"]["state"] == "s2"
    assert result["initial_state"] == "State s2: 0.7"
    assert result["state_values"]["s0"] == 0.2
