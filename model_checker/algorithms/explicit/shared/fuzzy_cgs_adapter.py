"""Turn a VITAMIN fuzzyCGS into a `fuzzy_sl.WCGS`, the model of ATL_STAR_F and SL1G_F.

- Joint actions and wildcards are read as ATL_STAR reads them (`ATL_STAR/cgs_adapter.py`).
- A label is the float FuzzyCGS read, taken as the decimal it prints as, so `0.2` is
  exactly 1/5 and not the binary double closest to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import cast

from model_checker.algorithms.explicit.ATL_STAR import cgs_adapter
from model_checker.fuzzy_sl import WCGS, to_value
from model_checker.fuzzy_sl.model import JointAction, State
from model_checker.parsers.game_structures.cgs.protocols import CGSProtocol


def adapt(cgs: CGSProtocol) -> WCGS:
    """The `WCGS` of `cgs`: agents `1..n`, the CGS's joint actions and its labels as
    exact values.

    Raises:
        ValueError: a joint action leads to two states, a state has no move, or a
            label is outside [0, 1].
    """
    ts = cgs_adapter.adapt(cgs).transition_system
    agents = tuple(range(1, cgs.get_number_of_agents() + 1))

    transitions: dict[tuple[State, JointAction], State] = {}
    for (source, symbol), targets in ts.transitions.items():
        per_agent = dict(cast("frozenset[tuple[int, str]]", symbol))
        joint_action = tuple(per_agent[agent] for agent in agents)
        if len(targets) != 1:
            raise ValueError(
                f"joint action {''.join(joint_action)!r} at state {source!r} leads to "
                f"{sorted(map(str, targets))}; it must lead to one state"
            )
        (target,) = targets
        transitions[(source, joint_action)] = target

    props = [str(p) for p in cgs.atomic_propositions]
    weights: dict[State, Mapping[str, object]] = {
        str(cgs.get_state_name_by_index(index)): {
            prop: _exact(value) for prop, value in zip(props, row, strict=True)
        }
        for index, row in enumerate(cgs.matrix_prop)
    }
    return WCGS(agents, str(cgs.initial_state), transitions, weights)


def _exact(label: object) -> Fraction:
    # str() of a float is the shortest decimal that reads back as it, i.e. what the
    # model file says; to_value(float) would keep the binary approximation.
    return to_value(str(label))
