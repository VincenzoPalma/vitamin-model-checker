"""Translates a `Coalition`/`Quantify` node into `automata.BlockSpec`:
which agents move in which decision block, and who owns each block.

`Coalition{A}` is ATL*[F]'s 2-block special case (the coalition, then
everyone else); `Quantify` is SL1G[F]'s general alternating prefix, one
block per quantifier (consecutive blocks with the same owner are merged --
`concurrent_to_turnbased_blocks` allows it, and it's free to do).
"""

from __future__ import annotations

from model_checker.automata import BlockSpec

from .formula import AgentId, Coalition, Quantifier, Quantify


def to_block_spec(
    formula: Coalition | Quantify, all_agents: frozenset[AgentId]
) -> BlockSpec:
    """Build the `BlockSpec` for `formula` against a model whose agents are
    `all_agents`.

    `Quantify`'s own binding prefix is already a bijection onto every
    agent, so it needs no outside information; `Coalition` only
    names its own agents, so `all_agents` supplies the rest (the opposing
    team).

    Raises:
        ValueError: `Coalition.agents` isn't a subset of `all_agents`, or
            `Quantify.bindings` doesn't bind exactly `all_agents`.
    """
    match formula:
        case Coalition(agents=agents):
            unknown = agents - all_agents
            if unknown:
                raise ValueError(
                    f"coalition names agent(s) not in the model: {sorted(map(str, unknown))}"
                )
            return [(0, frozenset(agents)), (1, frozenset(all_agents - agents))]
        case Quantify(prefix=prefix, bindings=bindings):
            agent_of_variable = {variable: agent for agent, variable in bindings}
            bound_agents = frozenset(agent_of_variable.values())
            if bound_agents != all_agents:
                raise ValueError(
                    f"quantifier bindings must cover exactly the model's agents; "
                    f"bound={sorted(map(str, bound_agents))}, model={sorted(map(str, all_agents))}"
                )
            raw_blocks = [
                (
                    0 if quantifier is Quantifier.EXISTS else 1,
                    frozenset({agent_of_variable[variable]}),
                )
                for quantifier, variable in prefix
            ]
            return _merge_consecutive(raw_blocks)
    raise TypeError(f"not a quantification node: {formula!r}")


def _merge_consecutive(raw_blocks: list[tuple[int, frozenset]]) -> BlockSpec:
    merged: BlockSpec = []
    for owner, agents in raw_blocks:
        if merged and merged[-1][0] == owner:
            merged[-1] = (owner, merged[-1][1] | agents)
        else:
            merged.append((owner, agents))
    return merged
