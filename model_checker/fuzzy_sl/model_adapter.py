"""Adapts a `WCGS` into `automata`'s `TransitionSystem` protocol.

Unlike `ATL_STAR/cgs_adapter.py` (VITAMIN's `CGSProtocol`, dense matrix,
compact/`|`-separated joint-action cells, wildcards), a `WCGS` is already
ours: states are already derived, actions are already per-agent and already
enumerable via `available_actions`. The only real conversion is shape --
`WCGS`'s positional `JointAction` tuple (aligned with `model.agents`) into
`automata`'s `frozenset[(agent, action)]` pairs.
"""

from __future__ import annotations

from model_checker.automata import TransitionSystem

from .model import WCGS


def adapt(model: WCGS) -> TransitionSystem:
    """Adapt `model` into a `TransitionSystem` with the same states and
    transitions, joint actions reshaped for `automata`.

    A `WCGS` cannot have a deadlocked state (checked in its own
    constructor), so every state here has at least one outgoing transition.
    """
    ts = TransitionSystem()
    for state in model.states:
        ts.add_state(state, initial=state == model.initial_state)

    for state in model.states:
        for joint_action in model.available_actions(state):
            symbol = frozenset(zip(model.agents, joint_action, strict=True))
            target = model.successor(state, joint_action)
            ts.add_transition(state, symbol, target)

    return ts
