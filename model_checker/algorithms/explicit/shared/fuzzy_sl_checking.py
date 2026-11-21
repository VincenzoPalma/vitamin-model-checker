"""Model checking and strategy synthesis shared by ATL_STAR_F and SL1G_F, on a fuzzyCGS.

- The formula's value at every state is computed by `fuzzy_sl` (one parity game per
  candidate value, see `model_checker/fuzzy_sl/verifier.py`).
- The checking result is ATLF's: `res` lists `(state, value)` in the model's state
  order and `initial_state` gives the initial state's value, both as floats.
  `state_values` maps each state to its value as a float, `exact_values` to its
  exact value, such as `"1/5"`.
- Synthesis returns the strategy achieving the formula's value at a state, as
  `fuzzy_sl.export_strategy` builds it (see `model_checker/fuzzy_sl/strategy_export.py`),
  with `state_values` and `exact_values` as above and `satisfied` None: a value in
  [0, 1] is not true or false.
"""

from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

from model_checker.algorithms.explicit.shared import fuzzy_cgs_adapter
from model_checker.engine.execution import execute_model_checking_with_parser
from model_checker.fuzzy_sl import (
    WCGS,
    Coalition,
    Formula,
    Prop,
    Quantify,
    export_strategy,
    sat,
    standard_registry,
    witness,
)
from model_checker.fuzzy_sl.formula import walk
from model_checker.parsers.formula_parser_factory import FormulaParserFactory
from model_checker.parsers.game_structures.cgs.protocols import CGSProtocol
from model_checker.utils.error_handler import (
    create_error_response,
    undeclared_propositions_message,
)


class _ErrorResponse(Exception):
    """Carries a VITAMIN error response out of `_parse_and_adapt`."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(response["res"])
        self.response = response


def _parse_and_adapt(
    logic_type: str, cgs: CGSProtocol, formula: str
) -> tuple[Formula, WCGS]:
    """Parse `formula` with `logic_type`'s parser, check its propositions, adapt `cgs`."""
    parser = FormulaParserFactory.get_parser_instance(logic_type)
    parsed = parser.parse(formula, n_agent=cgs.get_number_of_agents())
    if parsed is None:
        error_msg = parser.errors[0] if parser.errors else "Syntax error in formula"
        raise _ErrorResponse(create_error_response("syntax", error_msg))

    # fuzzy_sl reads an unknown proposition as 0 everywhere, so a typo would
    # still give a value.
    declared = {str(p) for p in cgs.atomic_propositions}
    undeclared = {node.name for node in walk(parsed) if isinstance(node, Prop)}
    undeclared -= declared
    if undeclared:
        raise _ErrorResponse(
            create_error_response(
                "semantic", undeclared_propositions_message(undeclared, declared)
            )
        )

    try:
        return parsed, fuzzy_cgs_adapter.adapt(cgs)
    except ValueError as e:
        raise _ErrorResponse(create_error_response("semantic", str(e))) from e


def core_checking(logic_type: str) -> Callable[[CGSProtocol, str], dict[str, Any]]:
    """The core checking function of `logic_type`, for `create_model_checking_entry`."""

    def _core(cgs: CGSProtocol, formula: str) -> dict[str, Any]:
        try:
            parsed, model = _parse_and_adapt(logic_type, cgs, formula)
            values = sat(parsed, model, standard_registry())
        except _ErrorResponse as e:
            return e.response
        except ImportError as e:
            return create_error_response("environment", str(e))
        except ValueError as e:
            return create_error_response("semantic", str(e))

        initial_state = str(cgs.initial_state)
        return {
            "res": f"Result: {[(state, float(values[state])) for state in _states(cgs)]}",
            "initial_state": f"Initial state {initial_state}: {float(values[initial_state])}",
            **_value_fields(cgs, values),
        }

    return _core


def _states(cgs: CGSProtocol) -> list[str]:
    return [str(cgs.get_state_name_by_index(i)) for i in range(len(cgs.states))]


def _value_fields(cgs: CGSProtocol, values: Mapping[Any, Fraction]) -> dict[str, Any]:
    """`state_values` (floats) and `exact_values` (fractions as strings), in state order."""
    states = _states(cgs)
    return {
        "state_values": {state: float(values[state]) for state in states},
        "exact_values": {state: str(values[state]) for state in states},
    }


def synthesize_strategy(
    logic_type: str, formula: str, filename: str, state: str | None = None
) -> dict[str, Any]:
    """A JSON-ready strategy achieving `formula`'s value on the model in `filename`
    at `state` (default: the initial state). The formula must be one coalition or one
    quantifier prefix, with none nested inside."""

    def _core(cgs: CGSProtocol, formula_text: str) -> dict[str, Any]:
        try:
            parsed, model = _parse_and_adapt(logic_type, cgs, formula_text)
        except _ErrorResponse as e:
            return e.response
        if not isinstance(parsed, Coalition | Quantify):
            return create_error_response(
                "semantic",
                "Strategy synthesis needs the formula to be one coalition or one "
                "quantifier prefix over a path formula, such as <<1>> F p",
            )
        at = str(cgs.initial_state) if state is None else state
        if at not in model.states:
            return create_error_response(
                "semantic",
                f"Unknown state {at!r}; states: {sorted(map(str, model.states))}",
            )

        try:
            values = sat(parsed, model, standard_registry())
            strategy = export_strategy(
                witness(parsed, model, standard_registry(), state=at)
            )
        except ImportError as e:
            return create_error_response("environment", str(e))
        except ValueError as e:
            return create_error_response("semantic", str(e))

        value = strategy["value"]
        return {
            "res": f"Result: strategy synthesized, value {value}",
            "initial_state": f"State {at}: {float(Fraction(value))}",
            "satisfied": None,
            "strategy": strategy,
            **_value_fields(cgs, values),
        }

    return execute_model_checking_with_parser(
        formula, filename, logic_type=logic_type, core_checking_func=_core
    )
