"""SL1G[F] model checking on fuzzy concurrent game structures.

- One-goal strategy logic with values in [0, 1]: a quantifier prefix such as
  `E x A y: (x, 1)(y, 2)` binds every agent to a strategy variable, and the value is
  the goal's best value under that alternation of choices.
- Solved by one parity game per candidate value (see `shared/fuzzy_sl_checking.py`).
"""

from typing import Any

from model_checker.algorithms.explicit.shared import fuzzy_sl_checking
from model_checker.engine.execution import create_model_checking_entry

model_checking = create_model_checking_entry(
    "SL1G_F", fuzzy_sl_checking.core_checking("SL1G_F")
)


def synthesize_strategy(
    formula: str, filename: str, state: str | None = None
) -> dict[str, Any]:
    """A JSON-ready strategy achieving `formula`'s value on the model in `filename` at
    `state` (default: the initial state). Only one quantifier prefix or coalition,
    with none nested inside, is supported; an existential agent's move can depend on
    the actions the universal agents quantified before it chose (`given`)."""
    return fuzzy_sl_checking.synthesize_strategy("SL1G_F", formula, filename, state)
