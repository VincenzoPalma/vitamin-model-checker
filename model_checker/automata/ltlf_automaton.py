"""Translate an LTL[F] formula plus a value threshold into a
deterministic, max-odd parity word automaton whose language is exactly
the computations whose LTL[F] value meets that threshold. Generalized to
weighted alphabets: each atomic proposition's value ranges over its own
finite `value_domain` (given by the caller), not a fixed `{0,1}`.

The automaton is built explicitly first (states = "consistent" functions
assigning a value to every subformula), then handed to Spot only for
determinization -- there is no LTL string Spot could translate directly,
but determinizing a hand-built automaton is exactly what
`Automaton.from_ltl` already asks Spot to do for a translated one.

Each state's outgoing edges all share one guard (`sigma` is a function of
the source state alone, never a free choice -- see
`_transition_allowed`'s docstring), built the same way `product.py`
builds a valuation guard: one Boolean "selector" proposition per
`(atom name, possible value)` pair, positive for the state's own value and
negated for every other value of that atom -- the finite-alphabet
analogue of `product.py`'s `valuation_bdd`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product as cartesian_product

from .automaton import Automaton, spot
from .ltlf import (
    Apply,
    Const,
    Formula,
    Next,
    Prop,
    Until,
    closure,
    free_nodes,
    value_sets,
)
from .quality import FunctionRegistry, Value

State = tuple[Value, ...]  # one value per free_nodes(formula), in that order


def _selector_ap(name: str, value: Value) -> str:
    """A Boolean proposition name standing for "atom `name` has value `value`"."""
    return f"__{name}_eq_{value.numerator}_{value.denominator}"


def _domain_aps(name: str, domain: Sequence[Value]) -> tuple[str, ...]:
    """The Boolean AP(s) encoding `name`'s finite value domain.

    A domain with at most two values (the common, purely-Boolean case --
    every atom of an unweighted LTL[F] formula) is encoded directly as the
    atom's own name: one literal, positive for the larger value and
    negated for the smaller. This keeps Boolean LTL[F] automata
    byte-for-byte comparable with Spot's own LTL translations (same AP
    names/polarity). A domain with 3+ values (the weighted case) needs a
    real one-hot encoding instead: no single literal can distinguish 3+
    possibilities.
    """
    if len(domain) <= 2:
        return (name,)
    return tuple(_selector_ap(name, value) for value in domain)


def true_aps(name: str, value: Value, domain: Sequence[Value]) -> tuple[str, ...]:
    """Which of `_domain_aps(name, domain)` are true when atom `name` has `value`.

    The other half of `threshold_automaton`'s encoding contract: use this
    (wrapped by `weighted_label`, or directly) to build `product()`'s
    `label` argument against a weighted transition system, so a
    `threshold_automaton` output can be `product()`-ed against a real
    model with no change to `product()` itself -- it already accepts an
    arbitrary AP-name `label` callable.
    """
    if len(domain) <= 2:
        return (name,) if value == max(domain) else ()
    return (_selector_ap(name, value),)


def _atom_literals(name: str, value: Value, domain: Sequence[Value]) -> tuple[str, ...]:
    """Every literal (one per `_domain_aps(name, domain)`, positive or
    negated) pinning atom `name` to exactly `value` -- a 3+-valued domain
    needs its *other* selectors asserted false too, not just the true one,
    or the guard would also match letters with an ambiguous/impossible
    combination of selectors true at once."""
    true = set(true_aps(name, value, domain))
    # Quoted, so Spot doesn't read `Granted` as `G ranted`.
    return tuple(
        f'"{ap}"' if ap in true else f'!"{ap}"' for ap in _domain_aps(name, domain)
    )


def weighted_label(
    labeling: Callable[[Hashable], Mapping[str, Value]],
    value_domains: Mapping[str, Sequence[Value]],
) -> Callable[[Hashable, Hashable], Iterable[str]]:
    """Build a `product()` `label` callable from a transition system's
    real-valued state labeling.

    `product()` calls `label(source, symbol)`; `symbol` is ignored here
    since a weighted atom's value comes from the *state*, not the
    transition taken to leave it (`[[pi, p]] = L(pi_0)(p)`).

    Args:
        labeling: a transition-system state's atom values, e.g. a WCGS
            adapter's `L(state)`.
        value_domains: the same domains passed to `threshold_automaton`.
    """

    def label(source: Hashable, symbol: Hashable) -> Iterable[str]:
        del symbol
        values = labeling(source)
        result: list[str] = []
        for name, domain in value_domains.items():
            result.extend(true_aps(name, values[name], domain))
        return result

    return label


def _evaluate_all(
    formula_closure: Sequence[Formula],
    free_values: Mapping[Formula, Value],
    registry: FunctionRegistry,
) -> dict[Formula, Value]:
    """Extend a free-node assignment to a full consistent state `g: cl(formula) -> Value`.

    `formula_closure` must be `ltlf.closure(formula)`'s own bottom-up order
    (not reversed) -- see that function's docstring for why reversing a
    pre-order dedup is unsafe when a subformula recurs at different depths.
    """
    values: dict[Formula, Value] = dict(free_values)
    for node in formula_closure:
        if isinstance(node, Const):
            values[node] = node.value
        elif isinstance(node, Apply):
            values[node] = registry.apply(
                node.function, [values[arg] for arg in node.args]
            )
        # Prop/Next/Until are already in free_values -- nothing to do.
    return values


def _consistent_states(
    free: Sequence[Formula], domains: Mapping[Formula, frozenset[Value]]
) -> Sequence[State]:
    """Every consistent state, as a tuple of values positionally aligned with `free`."""
    return list(cartesian_product(*(sorted(domains[node]) for node in free)))


def _transition_allowed(
    formula_closure: Sequence[Formula],
    values: Mapping[Formula, Value],
    successor_values: Mapping[Formula, Value],
) -> bool:
    """Whether `successor_values` may follow `values` (the transition relation `delta`).

    `sigma` itself isn't checked here: it's `{p in AP : values[p] == ...}`
    by construction, forced by `values` alone, so the caller only ever
    offers `successor_values` candidates under `values`'s own letter,
    see `threshold_automaton`, which builds every state's outgoing edges
    under that one fixed guard.
    """
    for node in formula_closure:
        if isinstance(node, Next):
            if values[node] != successor_values[node.operand]:
                return False
        elif isinstance(node, Until):
            left_v, right_v = values[node.left], values[node.right]
            if values[node] != max(right_v, min(left_v, successor_values[node])):
                return False
    return True


def threshold_automaton(
    formula: Formula,
    value_domains: Mapping[str, frozenset[Value] | set[Value]],
    registry: FunctionRegistry,
    threshold: Value,
) -> Automaton:
    """Build a deterministic, max-odd parity automaton accepting exactly the
    computations `pi` with `[[pi, formula]] >= threshold`.

    Args:
        value_domains: `V(p)` for every `Prop` name `p` in `formula` -- in
            model checking, the finite set of values that atom actually
            takes across the model's states.
        threshold: the predicate is `[threshold, 1]`.

    Raises:
        ImportError: if Spot isn't importable.
        KeyError: if some `Prop` in `formula` is missing from `value_domains`.
        ValueError: if `threshold` isn't in [0, 1].
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if spot is None:
        raise ImportError(
            "Spot is required for threshold_automaton() "
            "(pip install spottl on Linux, or conda-forge elsewhere; see docs/algorithms/atl_star.md)"
        )

    formula_closure = closure(formula)
    domains = value_sets(formula, value_domains, registry)
    free = free_nodes(formula)
    until_nodes = tuple(node for node in formula_closure if isinstance(node, Until))
    atom_names = tuple(
        dict.fromkeys(node.name for node in formula_closure if isinstance(node, Prop))
    )

    states = _consistent_states(free, domains)
    full_values = [
        _evaluate_all(formula_closure, dict(zip(free, state, strict=True)), registry)
        for state in states
    ]

    atom_domains = {name: tuple(sorted(value_domains[name])) for name in atom_names}

    # `spot.make_twa_graph()`'s own default dict, not an explicit
    # `spot.make_bdd_dict()`, confirmed empirically that the latter makes
    # the resulting automaton incomparable (`are_equivalent` raises "should
    # share their bdd_dict") against anything built via a different dict,
    # including ordinary `spot.translate(...)` output; the default dict
    # doesn't have this problem.
    aut = spot.make_twa_graph()
    bdict = aut.get_dict()
    for name in atom_names:
        for ap_name in _domain_aps(name, atom_domains[name]):
            aut.register_ap(spot.formula(f'"{ap_name}"'))

    if until_nodes:
        aut.set_generalized_buchi(len(until_nodes))
    else:
        aut.set_acceptance(0, "t")

    aut.new_states(len(states) + 1)  # +1 for the dedicated Q0 init state, placed last

    def guard_for(values: Mapping[Formula, Value]):
        literals = [
            literal
            for name in atom_names
            for literal in _atom_literals(name, values[Prop(name)], atom_domains[name])
        ]
        formula_str = " & ".join(literals) if literals else "1"
        return spot.formula_to_bdd(spot.formula(formula_str), bdict, aut)

    guards = [guard_for(values) for values in full_values]

    for i, values_i in enumerate(full_values):
        marks = [
            j
            for j, node in enumerate(until_nodes)
            if values_i[node.right] == values_i[node]
        ]
        for j, values_j in enumerate(full_values):
            if _transition_allowed(formula_closure, values_i, values_j):
                aut.new_edge(i, j, guards[i], marks)

    # No qualifying state is legitimate (the value set over-approximates the
    # values actually reachable, e.g. q & !q): `init_state` then has no
    # outgoing edges and the language is empty.
    init_candidates = [
        i for i, values in enumerate(full_values) if values[formula] >= threshold
    ]
    # NOT games/solver.py's anchor-state trick, an unmarked bridging edge
    # is harmless for a *game* (no word being read) but wrong here: it
    # would silently accept any first letter and shift every acceptance
    # check by one position. An NBA/NGBA's Q0 needs no bridge at all: the
    # first transition is simply chosen from any g in Q0's own outgoing
    # edges, so those edges (guard, target, marks alike) are copied onto
    # a state instead -- a *dedicated, fresh* one (`init_state`, reserved
    # above), not one of Q0's own members picked as a "representative".
    # An earlier version reused a member for this, which is wrong: that
    # member is typically also reachable later as an ordinary successor,
    # at which point it would wrongly keep carrying every other Q0
    # member's transitions too, not just its own, forever -- confirmed to
    # actually change the accepted language (not just look impure) via
    # `(p U q) | (r U s)`, whose `threshold_automaton` disagreed with
    # `spot.translate`'s own equivalent LTL translation under the old
    # approach. A fresh state has no incoming edges at all, so it can
    # never be revisited this way.
    init_state = len(states)
    for member in init_candidates:
        for edge in list(aut.out(member)):
            aut.new_edge(init_state, edge.dst, edge.cond, list(edge.acc.sets()))
    aut.set_init_state(init_state)

    det = spot.postprocess(aut, "parity", "deterministic")
    det = spot.change_parity(det, spot.parity_kind_max, spot.parity_style_odd)
    return Automaton(graph=det)
