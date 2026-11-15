"""Weighted concurrent game structures: the models of ATL*[F].

A WCGS is a concurrent game structure whose labelling maps each state to a
value in [0,1] per atomic proposition, rather than to a set of propositions
that hold.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from .values import ONE, ZERO, Value, to_value

AgentId = Hashable
State = Hashable
Action = Hashable
JointAction = tuple[Action, ...]

# Shared empty defaults, so the lookups below don't allocate on every miss.
_NO_ACTIONS: frozenset[JointAction] = frozenset()
_NO_WEIGHTS: dict[str, Value] = {}


class WCGS:
    """A weighted concurrent game structure.

    Args:
        agents: the agents, in the order joint actions are written in.
        initial_state: must have at least one outgoing transition.
        transitions: `(source, joint_action) -> target`.
        weights: `state -> {proposition: value}`; anything unlisted is 0.

    Raises:
        ValueError: on duplicate agents, a joint action of the wrong length, a
            weight outside [0,1], a weight on an unknown state, or a state with
            no successor.
    """

    def __init__(
        self,
        agents: Sequence[AgentId],
        initial_state: State,
        transitions: Mapping[tuple[State, JointAction], State],
        weights: Mapping[State, Mapping[str, object]] | None = None,
    ) -> None:
        self._agents = tuple(agents)
        if len(set(self._agents)) != len(self._agents):
            raise ValueError(f"agents contain duplicates: {self._agents!r}")
        self._agent_index = {agent: index for index, agent in enumerate(self._agents)}

        self._transitions: dict[tuple[State, JointAction], State] = {}
        by_source: dict[State, set[JointAction]] = {}
        for (source, joint_action), target in transitions.items():
            action = tuple(joint_action)
            if len(action) != len(self._agents):
                raise ValueError(
                    f"joint action {action!r} at state {source!r} has {len(action)} "
                    f"entries, expected {len(self._agents)}"
                )
            self._transitions[(source, action)] = target
            by_source.setdefault(source, set()).add(action)

        # Frozen up front so `available_actions` can hand them out without copying.
        self._by_source: dict[State, frozenset[JointAction]] = {
            source: frozenset(actions) for source, actions in by_source.items()
        }
        self._initial_state = initial_state
        self._states = frozenset(
            {initial_state, *self._by_source, *self._transitions.values()}
        )

        self._weights: dict[State, dict[str, Value]] = {}
        for state, assignment in (weights or {}).items():
            if state not in self._states:
                raise ValueError(f"weights given for unknown state {state!r}")
            self._weights[state] = {
                proposition: to_value(value)
                for proposition, value in assignment.items()
            }

        deadlocked = self._states - self._by_source.keys()
        if deadlocked:
            raise ValueError(
                f"states with no successor: {sorted(deadlocked, key=repr)}"
            )

        self._atomic_propositions = frozenset(
            proposition
            for assignment in self._weights.values()
            for proposition in assignment
        )
        # {0,1} is a hypothesis of the finiteness argument for satisfaction
        # values, not just whatever happens to occur in the weights.
        self._value_domain = frozenset(
            {
                ZERO,
                ONE,
                *(
                    value
                    for assignment in self._weights.values()
                    for value in assignment.values()
                ),
            }
        )
        self._domain_of_cache: dict[str, frozenset[Value]] = {}

    @property
    def agents(self) -> tuple[AgentId, ...]:
        return self._agents

    @property
    def states(self) -> frozenset[State]:
        """Every state, derived from the transitions rather than declared."""
        return self._states

    @property
    def initial_state(self) -> State:
        return self._initial_state

    def agent_index(self, agent: AgentId) -> int:
        """The position `agent` occupies in a joint action.

        Raises:
            KeyError: if `agent` is not in this model.
        """
        try:
            return self._agent_index[agent]
        except KeyError:
            raise KeyError(
                f"{agent!r} is not an agent of this model; agents: {list(self._agents)}"
            ) from None

    def successor(self, state: State, joint_action: JointAction) -> State | None:
        """The state reached from `state`, or `None` if that joint action is not
        available there.

        Raises:
            KeyError: if `state` is not in this model.
        """
        self._require_state(state)
        return self._transitions.get((state, tuple(joint_action)))

    def available_actions(self, state: State) -> frozenset[JointAction]:
        """The joint actions the model really offers at `state`.

        Raises:
            KeyError: if `state` is not in this model.
        """
        self._require_state(state)
        return self._by_source.get(state, _NO_ACTIONS)

    def weight(self, state: State, proposition: str) -> Value:
        """The degree to which `proposition` holds at `state`; 0 if unlisted.

        Raises:
            KeyError: if `state` is not in this model.
        """
        self._require_state(state)
        return self._weights.get(state, _NO_WEIGHTS).get(proposition, ZERO)

    def atomic_propositions(self) -> frozenset[str]:
        return self._atomic_propositions

    def value_domain(self) -> frozenset[Value]:
        """The model's contribution to the set of possible satisfaction values:
        the weights that occur, plus 0 and 1.

        Constants appearing in a formula can widen this further, so a caller
        working per-formula has to union the two.
        """
        return self._value_domain

    def value_domain_of(self, proposition: str) -> frozenset[Value]:
        """`proposition`'s own value domain: the weights it actually takes
        across this model's states, plus 0 and 1 -- narrower than
        `value_domain()`, which mixes in every other proposition's values
        too. Scoping a proposition's domain to only what it can really take
        (instead of the whole model's/formula's shared domain) keeps
        `threshold_automaton`'s state space from growing with *other*
        propositions' richness it has nothing to do with.

        Cached after the first call per `proposition`: a `WCGS` never
        changes after construction, and this gets called more than once
        for the same name within a single `sat()`/`witness()` query (once
        building `threshold_automaton`'s per-proposition domains, again
        inside `value_set`/`value_set_labelings`'s own `Prop` case).
        """
        cached = self._domain_of_cache.get(proposition)
        if cached is None:
            cached = frozenset(
                {
                    ZERO,
                    ONE,
                    *(self.weight(state, proposition) for state in self._states),
                }
            )
            self._domain_of_cache[proposition] = cached
        return cached

    def _require_state(self, state: State) -> None:
        if state not in self._states:
            raise KeyError(f"{state!r} is not a state of this model")

    def __repr__(self) -> str:
        return (
            f"WCGS(agents={list(self._agents)}, states={len(self._states)}, "
            f"transitions={len(self._transitions)}, initial_state={self._initial_state!r})"
        )
