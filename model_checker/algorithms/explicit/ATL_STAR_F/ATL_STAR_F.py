"""ATL*[F] model checking on fuzzy concurrent game structures.

- ATL* with values in [0, 1]: a formula's value at a state is the best value the
  coalition can guarantee, whatever the other agents do.
- The same algorithm as SL1G_F, of which ATL*[F] is the fragment with coalitions only
  (see `shared/fuzzy_sl_checking.py`).
"""

from model_checker.algorithms.explicit.shared.fuzzy_sl_checking import core_checking
from model_checker.engine.execution import create_model_checking_entry

model_checking = create_model_checking_entry("ATL_STAR_F", core_checking("ATL_STAR_F"))
