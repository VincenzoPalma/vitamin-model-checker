"""Model checking shared by ATL_STAR_F and SL1G_F, on a fuzzyCGS.

- The formula's value at every state is computed by `fuzzy_sl` (one parity game per
  candidate value, see `model_checker/fuzzy_sl/verifier.py`).
- The result is ATLF's: `res` lists `(state, value)` in the model's state order and
  `initial_state` gives the initial state's value, both as floats. `exact_values`
  maps each state to its exact value, such as `"1/5"`.
"""

from collections.abc import Callable
from typing import Any

from model_checker.algorithms.explicit.shared import fuzzy_cgs_adapter
from model_checker.fuzzy_sl import Prop, sat, standard_registry
from model_checker.fuzzy_sl.formula import walk
from model_checker.parsers.formula_parser_factory import FormulaParserFactory
from model_checker.parsers.game_structures.cgs.protocols import CGSProtocol
from model_checker.utils.error_handler import (
    create_error_response,
    undeclared_propositions_message,
)


def core_checking(logic_type: str) -> Callable[[CGSProtocol, str], dict[str, Any]]:
    """The core checking function of `logic_type`, for `create_model_checking_entry`."""

    def _core(cgs: CGSProtocol, formula: str) -> dict[str, Any]:
        parser = FormulaParserFactory.get_parser_instance(logic_type)
        parsed = parser.parse(formula, n_agent=cgs.get_number_of_agents())
        if parsed is None:
            error_msg = parser.errors[0] if parser.errors else "Syntax error in formula"
            return create_error_response("syntax", error_msg)

        # fuzzy_sl reads an unknown proposition as 0 everywhere, so a typo would
        # still give a value.
        declared = {str(p) for p in cgs.atomic_propositions}
        undeclared = {
            node.name for node in walk(parsed) if isinstance(node, Prop)
        } - declared
        if undeclared:
            return create_error_response(
                "semantic", undeclared_propositions_message(undeclared, declared)
            )

        try:
            model = fuzzy_cgs_adapter.adapt(cgs)
            values = sat(parsed, model, standard_registry())
        except ImportError as e:
            return create_error_response("environment", str(e))
        except ValueError as e:
            return create_error_response("semantic", str(e))

        states = [str(cgs.get_state_name_by_index(i)) for i in range(len(cgs.states))]
        initial_state = str(cgs.initial_state)
        return {
            "res": f"Result: {[(state, float(values[state])) for state in states]}",
            "initial_state": f"Initial state {initial_state}: {float(values[initial_state])}",
            "exact_values": {state: str(values[state]) for state in states},
        }

    return _core
