"""The SL1G[F] elimination algorithm: reduces model checking a
`Coalition`/`Quantify` sentence to a threshold loop over the finite
candidate value set (computed by `automata.ltlf.value_sets`), one parity
game per candidate via `automata.threshold_automaton` and
`automata.weighted_label` (its weighted-alphabet support).
Sentences nested inside a path formula are eliminated bottom-up first,
each replaced by a fresh weighted proposition carrying its own
already-computed per-state value -- the real-valued generalization of the
Boolean fresh-proposition trick `ATL_STAR.verifier._eliminate` uses.

ATL*[F] needs no separate algorithm: `Coalition` is SL1G[F]'s 2-block
special case, handled by the same code as `Quantify`.

`Witness` mirrors `ATL_STAR.verifier.Witness`, generalized from a bare
truth value to a real value everywhere: since every state has *some*
achievable value in a quantitative logic (there's no "doesn't hold" case
to restrict to), `_solve_sentence` computes a witness for every state
alongside its value, in the same threshold-descending pass -- eliminating
a nested sentence needs both anyway (see `_eliminate`), so computing them
together costs nothing extra and avoids re-solving the same games twice.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from itertools import count

from model_checker import automata

from .blocks import to_block_spec
from .formula import (
    Apply,
    Coalition,
    Const,
    Formula,
    Next,
    Prop,
    Quantify,
    Until,
    validate,
)
from .model import WCGS
from .model_adapter import adapt
from .parser import parse
from .values import FunctionRegistry, Value, standard_registry, to_value

State = Hashable


@dataclass(frozen=True)
class Witness:
    """A synthesized strategy witnessing a `Coalition`/`Quantify` sentence's
    own value at a particular state -- whatever `sat()` reports there.

    `strategy` is the raw, block-arena-level strategy for owner 0's blocks
    (the existentially-quantified/coalition agents); `start` is the
    product state `(model_state, automaton_state)` to read it from -- the
    automaton state is the memory the sentence's temporal structure needs.
    `move(agent)` reads one agent's own action out of it. `value` is what
    the strategy guarantees from the state it was synthesized at (kept
    unchanged by `advance`).

    `nested` carries, for every `Coalition`/`Quantify` eliminated out of
    this sentence's own path formula, that inner sentence's own witness at
    every state (unlike `ATL_STAR.verifier.Witness`, not just states where
    it "holds" -- a quantitative sentence always has *some* achievable
    value, so there's no analogous restriction). Keyed first by an
    internal marker name (meaningless alone), then by state -- look it up
    via `nested_witnesses_at`.
    """

    strategy: automata.Strategy
    start: tuple[State, int]
    block_spec: automata.BlockSpec
    product_ts: automata.TransitionSystem
    value: Value
    nested: dict[str, dict[State, Witness]] = field(default_factory=dict)

    def move(self, agent: Hashable) -> Value | Hashable | None:
        """The action `agent` should play at this witness's state.

        `None` if the strategy has no recorded move for `agent` here (e.g.
        `agent` sits in one of owner 1's blocks, which this witness never
        records a move for).

        Raises:
            ValueError: `agent` doesn't appear in `block_spec` at all.
        """
        projected = automata.project_blocks(
            self.strategy, self.block_spec, [self.start], agent
        )
        return projected.move(self.start)

    def nested_witnesses_at(self, state: State) -> list[Witness]:
        """Every nested sentence's own witness that applies at `state`."""
        return [
            by_state[state] for by_state in self.nested.values() if state in by_state
        ]

    def advance(self, symbol: Hashable) -> Witness | None:
        """Follow this witness one real step under `symbol` (the full
        joint action that occurred), auto-switching into a nested
        sentence's own witness once exactly one becomes available.

        Returns:
            `None` if `symbol` has no successor from `start` in the
            underlying product (an unrecognized action, or `complete()`'s
            rejecting sink).

        Raises:
            ValueError: `symbol` has more than one possible successor, or
                the resulting state has more than one nested witness
                available at once. The latter is a real, expected
                consequence of every state always having a nested witness
                here (see `nested`'s docstring): it triggers whenever 2+
                sentences were eliminated from the same path formula, not
                only when they happen to coincide as in the Boolean case.
        """
        targets = self.product_ts.successors(self.start, symbol)
        if not targets:
            return None
        if len(targets) > 1:
            raise ValueError(
                f"{symbol!r} from {self.start!r} has {len(targets)} possible successors, not 1"
            )
        (next_state,) = targets
        if not isinstance(next_state, tuple):
            return None  # complete()'s rejecting sink
        nested_here = self.nested_witnesses_at(next_state[0])
        if len(nested_here) > 1:
            raise ValueError(
                f"{next_state[0]!r} has {len(nested_here)} nested witnesses available at once"
            )
        if nested_here:
            return nested_here[0]
        return replace(self, start=next_state)


def sat(
    formula: Formula, model: WCGS, registry: FunctionRegistry
) -> dict[State, Value]:
    """The exact value of `formula` at every state of `model`.

    Raises:
        ValueError: if `formula` isn't a well-formed state formula against
            `registry` (`formula.validate`), or a `Coalition`/`Quantify`
            names/binds an agent that isn't in `model`.
    """
    validate(formula, registry)
    ts_cache: list[automata.TransitionSystem] = []
    registry_cache: list[automata.quality.FunctionRegistry] = []
    return _sat(formula, model, registry, ts_cache, registry_cache)


def value_at(
    formula: Formula, model: WCGS, state: State, registry: FunctionRegistry
) -> Value:
    """The exact value of `formula` at `state`.

    Raises:
        ValueError: see `sat`.
        KeyError: if `state` isn't a state of `model`.
    """
    if state not in model.states:
        raise KeyError(
            f"{state!r} is not a state of this model; states: {sorted(map(repr, model.states))}"
        )
    return sat(formula, model, registry)[state]


def check(
    model: WCGS,
    formula_text: str,
    registry: FunctionRegistry | None = None,
    state: State | None = None,
) -> Value:
    """Parse `formula_text` and return its exact value at `state` (default:
    `model`'s own initial state).

    Convenience composing `parser.parse` and `value_at` -- call them
    separately instead when checking several formulas against the same
    model.

    Raises:
        ParseError: `formula_text` is malformed.
        ValueError: the parsed formula isn't a state formula, or a
            `Coalition`/`Quantify` in it names/binds an agent outside `model`.
        KeyError: `state` isn't a state of `model`.
    """
    registry = standard_registry() if registry is None else registry
    formula = parse(formula_text, registry)
    if state is None:
        state = model.initial_state
    return value_at(formula, model, state, registry)


def holds(
    model: WCGS,
    formula_text: str,
    threshold: Value,
    registry: FunctionRegistry | None = None,
    state: State | None = None,
) -> bool:
    """Whether `check(model, formula_text, registry, state)`'s value is
    `>= threshold` -- the decision-problem version of `check`.

    Raises:
        ValueError: see `check`; also if `threshold` is outside [0,1].
        ParseError, KeyError: see `check`.
    """
    return check(model, formula_text, registry, state) >= to_value(threshold)


def witness(
    formula: Coalition | Quantify,
    model: WCGS,
    registry: FunctionRegistry,
    state: State | None = None,
) -> Witness:
    """The strategy witnessing `formula`'s own value at `state` (default:
    `model`'s own initial state).

    Raises:
        ValueError: see `sat`.
        KeyError: `state` isn't a state of `model`.
    """
    if state is None:
        state = model.initial_state
    elif state not in model.states:
        raise KeyError(
            f"{state!r} is not a state of this model; states: {sorted(map(repr, model.states))}"
        )
    validate(formula, registry)
    ts_cache: list[automata.TransitionSystem] = []
    registry_cache: list[automata.quality.FunctionRegistry] = []
    _values, witnesses = _solve_sentence(
        formula, model, registry, ts_cache, registry_cache
    )
    return witnesses[state]


def _adapted(
    model: WCGS, ts_cache: list[automata.TransitionSystem]
) -> automata.TransitionSystem:
    """Adapt `model` at most once per top-level `sat()`/`witness()` call,
    shared (via `ts_cache`) across every `Coalition`/`Quantify` reached
    from it -- sibling sentences under the same `Apply` and nested ones
    alike. `adapt()` is a real, measured bottleneck at scale (see
    ATL_STAR's own benchmarking of the same adapter shape), so re-adapting
    the same model once per sentence would be wasted, avoidable work.
    """
    if not ts_cache:
        ts_cache.append(adapt(model))
    return ts_cache[0]


def _automata_registry(
    registry: FunctionRegistry, registry_cache: list[automata.quality.FunctionRegistry]
) -> automata.quality.FunctionRegistry:
    """Convert `registry` to `automata`'s own registry type at most once
    per top-level `sat()`/`witness()` call, shared (via `registry_cache`)
    across every `Coalition`/`Quantify` reached from it -- same caching
    pattern as `_adapted`, for the same reason: `_solve_sentence` runs once
    per nested sentence too, and `registry` never changes mid-query.
    """
    if not registry_cache:
        registry_cache.append(_to_automata_registry(registry))
    return registry_cache[0]


def _sat(
    formula: Formula,
    model: WCGS,
    registry: FunctionRegistry,
    ts_cache: list[automata.TransitionSystem],
    registry_cache: list[automata.quality.FunctionRegistry],
) -> dict[State, Value]:
    match formula:
        case Prop(name=name):
            return {state: model.weight(state, name) for state in model.states}
        case Const(value=value):
            return dict.fromkeys(model.states, value)
        case Apply(function=function, args=args):
            arg_values = [
                _sat(arg, model, registry, ts_cache, registry_cache) for arg in args
            ]
            return {
                state: registry.apply(function, [av[state] for av in arg_values])
                for state in model.states
            }
        case Coalition() | Quantify():
            values, _witnesses = _solve_sentence(
                formula, model, registry, ts_cache, registry_cache
            )
            return values
    raise TypeError(f"not a formula: {formula!r}")


def _solve_sentence(
    formula: Coalition | Quantify,
    model: WCGS,
    registry: FunctionRegistry,
    ts_cache: list[automata.TransitionSystem],
    registry_cache: list[automata.quality.FunctionRegistry],
) -> tuple[dict[State, Value], dict[State, Witness]]:
    """Every state's exact value for `formula`, and a witness achieving
    it, in one threshold-descending pass."""
    flat, extra, nested = _eliminate(
        formula.path_formula, model, registry, ts_cache, registry_cache
    )
    block_spec = to_block_spec(formula, frozenset(model.agents))
    am_registry = _automata_registry(registry, registry_cache)

    # Every model state needs a verdict, not just the model's own designated
    # initial state -- `product()`'s traversal only reaches what's reachable
    # from `initial_states`, so solve from all of them at once (`solve()` supports
    # multiple initial states for exactly this reason). Mirrors ATL_STAR's
    # own `_solve_coalition`.
    ts = _adapted(model, ts_cache)
    probe = replace(ts, initial_states=set(ts.states))

    used_names = {
        node.name
        for node in automata.ltlf.walk(flat)
        if isinstance(node, automata.ltlf.Prop)
    }
    # Each proposition's own domain (`model.value_domain_of`), not the whole
    # formula's shared one -- a genuinely Boolean atom wrapped in a rich
    # function (e.g. `avg` over several propositions) must not inherit that
    # function's own, much wider, image: `threshold_automaton`'s state space
    # is a product over every free node's domain, so handing every
    # proposition the same wide domain inflates it needlessly.
    value_domains = {
        name: (
            frozenset(extra[name].values())
            if name in extra
            else model.value_domain_of(name)
        )
        for name in used_names
    }

    def labeling(state: State) -> dict[str, Value]:
        return {
            name: (extra[name][state] if name in extra else model.weight(state, name))
            for name in used_names
        }

    label = automata.weighted_label(labeling, value_domains)

    # Candidates from `flat` under the exact domains `threshold_automaton`
    # gets, not from the original path formula: there a nested sentence
    # only contributes its own path formula's value set, a superset of the
    # values it really takes, and a candidate above all of those has no
    # consistent initial state (`threshold_automaton` raises).
    path_values = automata.ltlf.value_sets(flat, value_domains, am_registry)[flat]

    # Descending: winning regions are monotone in the threshold (forcing
    # value >= v forces >= v' for any v' <= v), so the first (highest)
    # candidate a state wins at is its exact value.
    candidates = sorted(path_values, reverse=True)
    remaining = set(model.states)
    values: dict[State, Value] = {}
    witnesses: dict[State, Witness] = {}
    for v in candidates:
        if not remaining:
            break
        aut = automata.threshold_automaton(
            flat, value_domains, am_registry, threshold=v
        )
        product_ts, objective = automata.product(aut, probe, label=label)
        product_ts, objective = automata.complete(product_ts, objective, probe)
        game = automata.concurrent_to_turnbased_blocks(product_ts, block_spec)
        game.objective = replace(
            objective,
            priorities=automata.remap_priorities_blocks(
                objective.priorities, block_spec
            ),
        )
        solution = automata.solve(game)

        aut_init = aut.graph.get_init_state_number()
        won_now = {
            state
            for state in remaining
            if (state, aut_init) in solution.winning_regions[0]
        }
        for state in won_now:
            values[state] = v
            witnesses[state] = Witness(
                strategy=solution.strategies[0],
                start=(state, aut_init),
                block_spec=block_spec,
                product_ts=product_ts,
                value=v,
                nested=nested,
            )
        remaining -= won_now

    if remaining:
        raise RuntimeError(
            f"no candidate value decided state(s) {sorted(map(repr, remaining))}; every state's game "
            "value is itself a candidate, so this should be unreachable"
        )
    return values, witnesses


def _eliminate(
    path_formula: Formula,
    model: WCGS,
    registry: FunctionRegistry,
    ts_cache: list[automata.TransitionSystem],
    registry_cache: list[automata.quality.FunctionRegistry],
) -> tuple[
    automata.ltlf.Formula,
    dict[str, dict[State, Value]],
    dict[str, dict[State, Witness]],
]:
    """Translate a path formula into a flat `automata.ltlf.Formula`,
    eliminating any nested `Coalition`/`Quantify` bottom-up into fresh
    weighted propositions (Thm 6.3's construction for `snd > 1`).

    Returns:
        The flat formula, `{fresh_name: {state: value}}`, and
        `{fresh_name: {state: witness}}` for every fresh proposition
        minted (both empty if `path_formula` has no nested sentences, i.e.
        `snd(formula) == 1`).
    """
    extra: dict[str, dict[State, Value]] = {}
    nested: dict[str, dict[State, Witness]] = {}
    fresh_names = count()
    # `Coalition`/`Quantify` are frozen dataclasses (structural equality),
    # so this also catches two independently-built but identical nested
    # sentences, not just the same object reused -- either way, solving
    # the same game twice for one elimination pass would be pure waste.
    solved: dict[Coalition | Quantify, str] = {}

    def go(node: Formula) -> automata.ltlf.Formula:
        match node:
            case Prop(name=name):
                return automata.ltlf.Prop(name)
            case Const(value=value):
                return automata.ltlf.Const(value)
            case Apply(function=function, args=args):
                return automata.ltlf.Apply(function, tuple(go(arg) for arg in args))
            case Next(operand=operand):
                return automata.ltlf.Next(go(operand))
            case Until(left=left, right=right):
                return automata.ltlf.Until(go(left), go(right))
            case Coalition() | Quantify():
                name = solved.get(node)
                if name is None:
                    values, witnesses = _solve_sentence(
                        node, model, registry, ts_cache, registry_cache
                    )
                    name = f"__sentence_{next(fresh_names)}"
                    extra[name] = values
                    nested[name] = witnesses
                    solved[node] = name
                return automata.ltlf.Prop(name)
        raise TypeError(f"not a formula: {node!r}")

    return go(path_formula), extra, nested


def _to_automata_registry(
    registry: FunctionRegistry,
) -> automata.quality.FunctionRegistry:
    """Copy every function of a `fuzzy_sl` registry into an
    `automata.quality.FunctionRegistry` -- the two are deliberately
    separate types (`automata` never depends on a downstream repo's own
    registry shape), so this is a thin, lossless translation."""
    result = automata.quality.FunctionRegistry()
    for name in registry.names():
        function = registry[name]
        result.register(name, function.fn, function.arity)
    return result
