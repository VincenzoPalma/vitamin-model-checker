"""SL1G[F] model checking on fuzzy concurrent game structures.

- One-goal strategy logic with values in [0, 1]: a quantifier prefix such as
  `E x A y: (x, 1)(y, 2)` binds every agent to a strategy variable, and the value is
  the goal's best value under that alternation of choices.
- Solved by one parity game per candidate value (see `shared/fuzzy_sl_checking.py`).
"""

from model_checker.algorithms.explicit.shared.fuzzy_sl_checking import core_checking
from model_checker.engine.execution import create_model_checking_entry

model_checking = create_model_checking_entry("SL1G_F", core_checking("SL1G_F"))
