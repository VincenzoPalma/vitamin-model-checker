"""SL1G[F] parser entry point for `vitamin.parsers`, also the base of ATL_STAR_F's.

- The parsing is `fuzzy_sl`'s hand-written parser (`model_checker/fuzzy_sl/parser.py`),
  whose syntax follows ATL_STAR's and NatSL's: `<<1>> F p`, `E x A y: (x, 1)(y, 2) F p`.
- On top of it, the model's agents: an agent is a number from 1 to `n_agent`, a
  coalition names at least one, and a quantifier prefix binds every agent once.
"""

from typing import Any

from model_checker.fuzzy_sl import (
    Coalition,
    Formula,
    ParseError,
    Quantify,
    parse,
    standard_registry,
)
from model_checker.fuzzy_sl.formula import walk


class _AgentError(ValueError):
    """A formula that parses but does not fit the model's agents."""


class SL1GFParser:
    """`parse` returns a `fuzzy_sl` formula, or None with the reason in `errors`."""

    allow_quantifiers = True

    def __init__(self) -> None:
        self.errors: list[str] = []

    def parse(self, formula: str, n_agent: int | None = None, **_: Any):
        """Parse `formula`, returning a `fuzzy_sl` formula or None (see `self.errors`)."""
        self.errors = []
        try:
            parsed = parse(formula, standard_registry())
            self._check_agents(parsed, n_agent)
            return parsed
        except (ParseError, _AgentError) as e:
            self.errors.append(str(e))
            return None
        except RecursionError:
            self.errors.append("Formula is nested too deeply to parse")
            return None

    def _check_agents(self, formula: Formula, n_agent: int | None) -> None:
        for node in walk(formula):
            if isinstance(node, Coalition):
                if not node.agents:
                    raise _AgentError("A coalition must name at least one agent")
                _check_range(node.agents, n_agent)
            elif isinstance(node, Quantify):
                if not self.allow_quantifiers:
                    raise _AgentError(
                        "Quantifiers are SL1G_F syntax: write a coalition <<...>> here"
                    )
                bound = [agent for agent, _variable in node.bindings]
                _check_range(bound, n_agent)
                if n_agent is not None and set(bound) != set(range(1, n_agent + 1)):
                    missing = sorted(set(range(1, n_agent + 1)) - set(bound))
                    raise _AgentError(
                        f"Agent(s) {missing} not bound: a quantifier prefix binds "
                        f"every agent from 1 to {n_agent} once"
                    )


def _check_range(agents, n_agent: int | None) -> None:
    for agent in sorted(agents, key=str):
        if not isinstance(agent, int):
            raise _AgentError(f"Agent {agent!r} is not a number: agents are 1, 2, ...")
        if agent < 1 or (n_agent is not None and agent > n_agent):
            raise _AgentError(f"Agent {agent} out of range [1, {n_agent}]")
