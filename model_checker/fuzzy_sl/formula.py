"""AST for SL1G[F], the one-goal fragment of SL[F] -- ATL*[F] is its
non-alternating special case (`Coalition`), kept as its own node rather than
desugared into `Quantify`, since a coalition doesn't bind every agent.

    phi ::= p | c | <<A>> psi | Q psi | f(phi, ..., phi)
    psi ::= phi | X psi | psi U psi | f(psi, ..., psi)
    Q   ::= (exists x | forall x)+ (agent, x)+     -- see Quantify
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from enum import Enum, auto

from .model import WCGS
from .values import FunctionRegistry, Value, to_value

AgentId = Hashable
Var = Hashable


@dataclass(frozen=True)
class Prop:
    """An atomic proposition; its value is read from the model's weights."""

    name: str


@dataclass(frozen=True)
class Const:
    """A literal value."""

    value: Value

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", to_value(self.value))


@dataclass(frozen=True)
class Apply:
    """A function of F applied to subformulas. Covers every connective."""

    function: str
    args: tuple[Formula, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True)
class Coalition:
    """`<<A>> psi`: the value A can force, maximized over A's strategies and
    minimized over everyone else's."""

    agents: frozenset[AgentId]
    path_formula: Formula

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", frozenset(self.agents))


class Quantifier(Enum):
    EXISTS = auto()
    FORALL = auto()


@dataclass(frozen=True)
class Quantify:
    """SL1G[F]'s one-goal sentence: a quantification prefix (existential or
    universal, one per variable, in order) immediately followed by a binding
    prefix, then a path formula with no further quantification of its own
    (nested `Quantify`/`Coalition` sentences are fine -- that's sentence
    nesting, not a second quantification inside this one's goal).

    `prefix`: `(Quantifier, variable)` pairs, in the order strategies become
    informed of each other -- this is the order `verifier.py` later hands to
    `automata`'s block-based arena unfolding.
    `bindings`: `(agent, variable)` pairs. Simplification over full SL[SG]:
    exactly one binding per prefix variable (a bijection), not the general
    "every agent bound, variables possibly shared across agents" case --
    covers every case actually needed and keeps this AST/parser
    unambiguous; revisit only if shared strategies are ever needed.

    Raises:
        ValueError: a repeated variable in `prefix`, a repeated agent in
            `bindings`, or `bindings` not using exactly `prefix`'s variables.
    """

    prefix: tuple[tuple[Quantifier, Var], ...]
    bindings: tuple[tuple[AgentId, Var], ...]
    path_formula: Formula

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefix", tuple(self.prefix))
        object.__setattr__(self, "bindings", tuple(self.bindings))

        prefix_vars = [variable for _quantifier, variable in self.prefix]
        if len(set(prefix_vars)) != len(prefix_vars):
            raise ValueError(
                f"quantification prefix has repeated variables: {prefix_vars!r}"
            )

        binding_agents = [agent for agent, _variable in self.bindings]
        if len(set(binding_agents)) != len(binding_agents):
            raise ValueError(f"binding prefix has repeated agents: {binding_agents!r}")

        binding_vars = [variable for _agent, variable in self.bindings]
        if set(binding_vars) != set(prefix_vars) or len(binding_vars) != len(
            prefix_vars
        ):
            raise ValueError(
                f"binding prefix must use exactly the quantified variables once each; "
                f"quantified={sorted(map(str, prefix_vars))}, bound={sorted(map(str, binding_vars))}"
            )


@dataclass(frozen=True)
class Next:
    operand: Formula


@dataclass(frozen=True)
class Until:
    left: Formula
    right: Formula


Formula = Prop | Const | Apply | Coalition | Quantify | Next | Until


def children(formula: Formula) -> tuple[Formula, ...]:
    """The immediate subformulas of `formula`."""
    match formula:
        case Prop() | Const():
            return ()
        case Apply(args=args):
            return args
        case Coalition(path_formula=body) | Quantify(path_formula=body):
            return (body,)
        case Next(operand=operand):
            return (operand,)
        case Until(left=left, right=right):
            return (left, right)
    raise TypeError(f"not a formula: {formula!r}")


def walk(formula: Formula) -> Iterator[Formula]:
    """Every subformula, `formula` itself included."""
    yield formula
    for child in children(formula):
        yield from walk(child)


def constants(formula: Formula) -> frozenset[Value]:
    """The literal values in `formula`, which widen a model's value domain."""
    return frozenset(node.value for node in walk(formula) if isinstance(node, Const))


def value_set(
    formula: Formula, model: WCGS, registry: FunctionRegistry
) -> frozenset[Value]:
    """The finite set of values `formula` can take against `model` under
    `registry`: a proposition contributes its own `model.value_domain_of`
    wherever it occurs (not the whole model's shared `value_domain()` --
    same reasoning as `_solve_sentence`'s own per-proposition domains:
    a narrower proposition must not inherit every other proposition's
    richness), a constant contributes itself, a function's value set is
    the image of its arguments' value sets (`registry.image`), and
    everything else -- temporal operators, `Coalition`, `Quantify` -- just
    recycles its operand(s)' values without creating new ones.
    """
    match formula:
        case Prop(name=name):
            return model.value_domain_of(name)
        case Const(value=value):
            return frozenset({value})
        case Apply(function=function, args=args):
            arg_sets = [value_set(arg, model, registry) for arg in args]
            return frozenset(registry.image(function, arg_sets))
        case Coalition(path_formula=body) | Quantify(path_formula=body):
            return value_set(body, model, registry)
        case Next(operand=operand):
            return value_set(operand, model, registry)
        case Until(left=left, right=right):
            return value_set(left, model, registry) | value_set(right, model, registry)
    raise TypeError(f"not a formula: {formula!r}")


def _closure(formula: Formula) -> tuple[Formula, ...]:
    """Every subformula of `formula`, each exactly once, in a bottom-up
    order (every node after all of its own children).

    A plain pre-order dedup, then reversed, looks like it gives the same
    order but doesn't: when the same subformula recurs at different depths,
    its shallowest occurrence can precede a node it is also a child of, so
    reversing leaves that parent before its own child. Recursing into
    children first, and recording a node only once every child already is,
    is correct regardless of sharing (same fix as `automata.ltlf.closure`).
    """
    seen: dict[Formula, None] = {}

    def visit(node: Formula) -> None:
        if node in seen:
            return
        for child in children(node):
            visit(child)
        seen[node] = None

    visit(formula)
    return tuple(seen)


def value_set_labelings(
    formula: Formula, model: WCGS, registry: FunctionRegistry
) -> Iterator[Value]:
    """`V(formula)`, enumerated one labeling at a time (Lemma 2.5 / section
    2.3.1's PSPACE algorithm) instead of materialized subformula-by-subformula
    the way `value_set` is: a labeling fixes one value for every "free" node
    -- a `Prop` (any value in its own domain) or an `Until` (whichever side's
    *already-fixed* value it picks) -- and forces everything else (`Const`,
    `Apply`, `Next`, `Coalition`, `Quantify`) deterministically from values
    already fixed lower in the formula. `frozenset(value_set_labelings(f, m,
    r)) == value_set(f, m, r)` for every `f`/`m`/`r` -- same result, reached
    without ever holding more than one labeling's values (one `Value` per
    subformula) at once.

    This buys space, not time, and the time cost is a real practical risk,
    confirmed (not just theoretical) by wiring this into `verifier._solve_sentence`
    once and measuring it on the existing nested-coalition-depth benchmark:
    `value_set`'s union-based approach collapses repeated structure for free
    (e.g. `Until`'s value set is `value_set(left) | value_set(right)` --
    unioning `{0,1}` into itself at every nesting level stays `{0,1}`,
    regardless of depth), while this function enumerates individual
    labelings and cannot exploit that collapse: a `nested_coalition_formula`-
    style formula (one `Until` per nesting level, from `F`'s desugaring)
    went from ~0.05s to several seconds already at depth 60 when this
    replaced `value_set` in that call site -- reverted the same session.
    Prefer `value_set` unless a caller specifically needs to avoid holding
    the full set at once *and* the formula's free nodes are shallow/few
    enough that this risk doesn't apply; profile before switching, don't
    assume parity.

    Yields the same value again whenever a different labeling forces it --
    wrap in `set(...)`/`frozenset(...)` for a deduplicated result.
    """
    order = _closure(formula)

    def go(index: int, values: dict[Formula, Value]) -> Iterator[Value]:
        if index == len(order):
            yield values[formula]
            return
        node = order[index]
        match node:
            case Const(value=value):
                values[node] = value
                yield from go(index + 1, values)
            case Prop(name=name):
                for candidate in model.value_domain_of(name):
                    values[node] = candidate
                    yield from go(index + 1, values)
            case Apply(function=function, args=args):
                values[node] = registry.apply(function, [values[arg] for arg in args])
                yield from go(index + 1, values)
            case Coalition(path_formula=body) | Quantify(path_formula=body):
                values[node] = values[body]
                yield from go(index + 1, values)
            case Next(operand=operand):
                values[node] = values[operand]
                yield from go(index + 1, values)
            case Until(left=left, right=right):
                for candidate in {values[left], values[right]}:
                    values[node] = candidate
                    yield from go(index + 1, values)
            case _:
                raise TypeError(f"not a formula: {node!r}")

    yield from go(0, {})


def is_state_formula(formula: Formula) -> bool:
    """Whether `formula` has a value at a state, with no path to read it along.

    Temporal operators do not; a coalition does, whatever its body. `Apply`
    inherits the answer from its arguments, which is what carries the
    restriction down into nested connectives.
    """
    match formula:
        case Prop() | Const() | Coalition() | Quantify():
            return True
        case Apply(args=args):
            return all(is_state_formula(arg) for arg in args)
        case Next() | Until():
            return False
    raise TypeError(f"not a formula: {formula!r}")


def validate(formula: Formula, registry: FunctionRegistry) -> None:
    """Check that `formula` is well formed against `registry`.

    Trees violating the two-sorted grammar are constructible, since a single
    Python type covers both sorts, so this has to be checked rather than assumed
    -- there is no verifier downstream that would otherwise catch it.

    Agents are not checked here: whether a coalition names agents that exist is
    a question about a model, not about the formula.

    Raises:
        ValueError: if the formula is a path formula, or names a function that
            is not in F or applies one at the wrong arity.
    """
    if not is_state_formula(formula):
        raise ValueError(
            "a formula must have a value at a state, but this one needs a path"
        )
    for node in walk(formula):
        if isinstance(node, Apply):
            if node.function not in registry:
                raise ValueError(
                    f"{node.function!r} is not in F; registered: {sorted(registry.names())}"
                )
            registry.require_arity(node.function, len(node.args))
