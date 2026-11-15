"""Turn a `Witness` into dicts and lists that `json.dumps` accepts.

The shape of ATL_STAR's export (`{start, moves, transitions}`, walked from `start`
so the memory's evolution can be replayed), adapted to block arenas and values:

- An existential block can come after a universal one (`A y E x: ...`), so its
  agents' moves depend on what the earlier blocks chose in the same round: each
  move lists those actions as `given` (always empty for a coalition, which moves
  first).
- `transitions` lists, per reachable position, every real successor of a joint
  action the strategy allows, so every choice of the universal agents is kept.
- `value` is the witnessed value as an exact fraction, such as `"4/5"`.
- Nested sentences are rejected: in a quantitative logic every state would carry a
  nested witness, so the export would need one per state.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypedDict, cast

from .verifier import Witness

JointAction = frozenset[tuple[Hashable, Hashable]]


class Position(TypedDict):
    state: Hashable
    memory: int


class StrategyMove(TypedDict):
    state: Hashable
    memory: int
    agent: Hashable
    action: Hashable
    given: list[tuple[Hashable, Hashable]]


class StrategyTransition(TypedDict):
    source: Position
    joint_action: list[tuple[Hashable, Hashable]]
    target: Position


class StrategyExport(TypedDict):
    value: str
    start: Position
    moves: list[StrategyMove]
    transitions: list[StrategyTransition]


def export_strategy(witness: Witness) -> StrategyExport:
    """Export the part of `witness`'s strategy that a run from `witness.start` can reach.

    Raises:
        ValueError: the witness has nested witnesses, or the strategy has a gap or
            reaches the rejecting sink, instead of returning an export that is
            silently incomplete.
    """
    if witness.nested:
        raise ValueError("export_strategy does not support nested sentences yet")

    outgoing: dict[Hashable, list[tuple[JointAction, Hashable]]] = {}
    for (source, symbol), targets in witness.product_ts.transitions.items():
        for target in targets:
            outgoing.setdefault(source, []).append((cast(JointAction, symbol), target))

    moves: dict[tuple[Hashable, JointAction, Hashable], Hashable] = {}
    transitions: list[StrategyTransition] = []
    seen: set[Hashable] = {witness.start}
    frontier: list[Hashable] = [witness.start]
    while frontier:
        position = frontier.pop()
        for joint_action, target in outgoing.get(position, []):
            if not _allowed(witness, position, joint_action, moves):
                continue
            reaches_sink = not isinstance(
                target, tuple
            )  # complete()'s sink is no (state, memory)
            if reaches_sink:
                raise ValueError(
                    f"the strategy reaches the rejecting sink from {position!r}"
                )
            transitions.append(
                {
                    "source": _position(position),
                    "joint_action": sorted(joint_action),
                    "target": _position(target),
                }
            )
            if target not in seen:
                seen.add(target)
                frontier.append(target)

    exported_moves: list[StrategyMove] = [
        {
            **_position(position),
            "agent": agent,
            "action": action,
            "given": sorted(given),
        }
        for (position, given, agent), action in moves.items()
    ]
    return {
        "value": str(witness.value),
        "start": _position(witness.start),
        "moves": sorted(exported_moves, key=repr),
        "transitions": sorted(transitions, key=repr),
    }


def _allowed(
    witness: Witness,
    position: Hashable,
    joint_action: JointAction,
    moves: dict[tuple[Hashable, JointAction, Hashable], Hashable],
) -> bool:
    """Whether the strategy can play `joint_action` at `position`: every existential
    block's part of it is the strategy's choice there, given the earlier blocks'
    parts. When it can, those choices are added to `moves`."""
    current: Hashable = position
    partial: JointAction = frozenset()
    chosen: list[tuple[JointAction, JointAction]] = []
    for owner, agents in witness.block_spec:
        step = frozenset(
            (agent, action) for agent, action in joint_action if agent in agents
        )
        if owner == 0:
            choice = witness.strategy.move(current)
            if choice is None:
                raise ValueError(
                    f"no strategy choice at reachable position {current!r}"
                )
            if choice != step:
                return False
            chosen.append((partial, step))
        partial = partial | step
        # The arena's intermediate state for this round so far.
        current = (position, partial)
    for given, step in chosen:
        for agent, action in step:
            moves[(position, given, agent)] = action
    return True


def _position(position: Hashable) -> Position:
    state, memory = cast("tuple[Hashable, int]", position)
    return {"state": state, "memory": memory}
