from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import TypeAlias, cast

from ..transition_system import TransitionSystem
from .game import Game
from .strategy import Strategy

# One (player, action) pair per agent, a "frozen dict".
JointAction: TypeAlias = frozenset[tuple[Hashable, Hashable]]

# Ordered decision blocks `(owner, agents)`, owner 0 or 1; every agent of the
# joint actions is in exactly one block.
BlockSpec: TypeAlias = list[tuple[int, frozenset[Hashable]]]

# (from_state, step_action, to_state, to_state's owner); the last two are None
# on the last block, whose step leads to the real target.
_Step: TypeAlias = tuple[Hashable, JointAction, "Hashable | None", "int | None"]


def concurrent_to_turnbased(
    ts: TransitionSystem, controlled_players: AbstractSet[Hashable]
) -> Game:
    """Turn a concurrent transition system into a turn-based 2-player game: player 0
    (`controlled_players`) picks its part of each move, then player 1 the rest.
    `objective` is left for the caller, through `remap_priorities`."""
    all_agents = _agents_of(ts)
    opponents = all_agents - controlled_players
    blocks: BlockSpec = [
        (0, frozenset(controlled_players)),
        (1, frozenset(opponents)),
    ]
    _validate_blocks(ts, blocks, all_agents=all_agents)
    return _build_turnbased(ts, blocks)


def concurrent_to_turnbased_blocks(ts: TransitionSystem, blocks: BlockSpec) -> Game:
    """Turn a concurrent transition system into a turn-based 2-player game, one
    decision block after the other: each block's owner picks the actions of its
    agents, knowing every earlier block's choice. `objective` is left for the
    caller, through `remap_priorities_blocks`.

    Raises:
        ValueError: an agent is in two blocks, or in none.
    """
    _validate_blocks(ts, blocks)
    return _build_turnbased(ts, blocks)


def _build_turnbased(ts: TransitionSystem, blocks: BlockSpec) -> Game:
    """The unfolding, for already validated `blocks`."""
    arena = TransitionSystem()
    player_states: dict[int, set[Hashable]] = {0: set(), 1: set()}

    first_owner = blocks[0][0]
    for state in ts.states:
        arena.add_state(state, initial=state in ts.initial_states)
        player_states[first_owner].add(state)

    for (source, symbol), targets in ts.transitions.items():
        joint_action = cast(JointAction, symbol)
        for from_state, step_action, to_state, next_owner in _unfold_steps(
            source, joint_action, blocks
        ):
            if to_state is None:
                for target in targets:
                    arena.add_transition(from_state, step_action, target)
            else:
                arena.add_transition(from_state, step_action, to_state)
                player_states[cast(int, next_owner)].add(to_state)

    return Game(arena=arena, player_states=player_states)


def _unfold_steps(
    source: Hashable, joint_action: JointAction, blocks: BlockSpec
) -> Iterator[_Step]:
    """One `_Step` per block for one joint action; an intermediate state is
    `(source, partial joint action chosen so far)`."""
    cur_state: Hashable = source
    partial: JointAction = frozenset()
    last = len(blocks) - 1
    for idx, (_owner, agents) in enumerate(blocks):
        step_action = frozenset(
            (player, action) for player, action in joint_action if player in agents
        )
        if idx < last:
            partial = partial | step_action
            next_state = (source, partial)
            yield cur_state, step_action, next_state, blocks[idx + 1][0]
            cur_state = next_state
        else:
            yield cur_state, step_action, None, None


def _agents_of(ts: TransitionSystem) -> set[Hashable]:
    """Every player in `ts`'s joint actions."""
    return {
        player
        for symbol in ts.alphabet
        for player, _action in cast(JointAction, symbol)
    }


def _validate_blocks(
    ts: TransitionSystem,
    blocks: BlockSpec,
    all_agents: set[Hashable] | None = None,
) -> None:
    """Check `blocks` partitions the agents; `all_agents` saves a rescan."""
    if not blocks:
        raise ValueError("blocks must be non-empty")
    seen: set[Hashable] = set()
    for owner, agents in blocks:
        if owner not in (0, 1):
            raise ValueError(f"block owner must be 0 or 1, got {owner!r}")
        overlap = seen & agents
        if overlap:
            raise ValueError(
                f"agent(s) {sorted(map(str, overlap))} appear in more than one block"
            )
        seen |= agents
    missing = (all_agents if all_agents is not None else _agents_of(ts)) - seen
    if missing:
        raise ValueError(
            f"agent(s) {sorted(map(str, missing))} appear in ts's joint actions "
            "but not in any block"
        )


def remap_priorities(
    priorities: dict[tuple[Hashable, Hashable, Hashable], int],
    controlled_players: AbstractSet[Hashable],
) -> dict[tuple[Hashable, Hashable, Hashable], int]:
    """Move the priorities of `ts` transitions onto the arena of
    `concurrent_to_turnbased`: only the second of the two arena steps keeps the
    color, so each real step still counts once."""
    all_players = {
        player
        for _source, symbol, _target in priorities
        for player, _action in cast(JointAction, symbol)
    }
    opponents = all_players - controlled_players
    blocks: BlockSpec = [
        (0, frozenset(controlled_players)),
        (1, frozenset(opponents)),
    ]
    return remap_priorities_blocks(priorities, blocks)


def remap_priorities_blocks(
    priorities: dict[tuple[Hashable, Hashable, Hashable], int],
    blocks: BlockSpec,
) -> dict[tuple[Hashable, Hashable, Hashable], int]:
    """Move the priorities of `ts` transitions onto the arena of
    `concurrent_to_turnbased_blocks`: only the last block's step keeps the color."""
    remapped: dict[tuple[Hashable, Hashable, Hashable], int] = {}
    for (source, symbol, target), priority in priorities.items():
        joint_action = cast(JointAction, symbol)
        *_, last_step = _unfold_steps(source, joint_action, blocks)
        from_state, step_action, _to_state, _next_owner = last_step
        remapped[(from_state, step_action, target)] = priority
    return remapped


def project_blocks(
    strategy: Strategy,
    blocks: BlockSpec,
    real_states: Iterable[Hashable],
    player: Hashable,
) -> Strategy:
    """`player`'s own actions at the real states, following the strategy through
    the earlier blocks' intermediate states. A state is left out when the strategy
    has no move somewhere on the way (e.g. its owner loses there).

    Raises:
        ValueError: `player` is in no block.
    """
    if not any(player in agents for _owner, agents in blocks):
        raise ValueError(f"{player!r} doesn't appear in any block")

    choices: dict[Hashable, Hashable] = {}
    for real_state in real_states:
        cur_state: Hashable = real_state
        partial: JointAction = frozenset()
        for _owner, agents in blocks:
            step_action = strategy.move(cur_state)
            if step_action is None:
                break
            step_dict = dict(cast(JointAction, step_action))
            if player in agents:
                if player in step_dict:
                    choices[real_state] = step_dict[player]
                break
            partial = partial | cast(JointAction, step_action)
            cur_state = (real_state, partial)

    return Strategy(choices)
