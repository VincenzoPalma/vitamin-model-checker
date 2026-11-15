"""ATL*[F] model checking on fuzzy concurrent game structures.

- ATL* with values in [0, 1]: a formula's value at a state is the best value the
  coalition can guarantee, whatever the other agents do.
- The same algorithm as SL1G_F, of which ATL*[F] is the fragment with coalitions only
  (see `shared/fuzzy_sl_checking.py`).
"""

from typing import Any

from model_checker.algorithms.explicit.shared import fuzzy_sl_checking
from model_checker.engine.execution import create_model_checking_entry

model_checking = create_model_checking_entry(
    "ATL_STAR_F", fuzzy_sl_checking.core_checking("ATL_STAR_F")
)


def synthesize_strategy(
    formula: str, filename: str, state: str | None = None
) -> dict[str, Any]:
    """A JSON-ready strategy achieving `formula`'s value on the model in `filename` at
    `state` (default: the initial state). Only one coalition, with none nested
    inside, is supported."""
    return fuzzy_sl_checking.synthesize_strategy("ATL_STAR_F", formula, filename, state)
