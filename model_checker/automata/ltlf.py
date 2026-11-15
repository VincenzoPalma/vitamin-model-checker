"""AST for LTL[F]: quantitative LTL where Boolean connectives are replaced
by arbitrary functions over [0,1].

Generic and logic-agnostic like the rest of this package: `Apply` names its
function by a plain string, resolved against whatever function registry a
caller supplies elsewhere -- this module has no registry of its own, and no
concept of strategies/coalitions/quantifiers (those are a client repo's
concern, e.g. `fuzzy_sl`'s `Coalition`/`Quantify`).

    phi ::= p | c | f(phi, ..., phi) | X phi | phi U phi
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from .quality import FunctionRegistry, Value, to_value


@dataclass(frozen=True)
class Prop:
    """An atomic proposition; its value is read from a model's weights."""

    name: str


@dataclass(frozen=True)
class Const:
    """A literal value in [0,1]."""

    value: Value

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", to_value(self.value))


@dataclass(frozen=True)
class Apply:
    """A named function applied to subformulas. Covers every connective."""

    function: str
    args: tuple[Formula, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True)
class Next:
    operand: Formula


@dataclass(frozen=True)
class Until:
    left: Formula
    right: Formula


Formula = Prop | Const | Apply | Next | Until


def children(formula: Formula) -> tuple[Formula, ...]:
    """The immediate subformulas of `formula`."""
    match formula:
        case Prop() | Const():
            return ()
        case Apply(args=args):
            return args
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


def closure(formula: Formula) -> tuple[Formula, ...]:
    """`cl(formula)`: each subformula once, in bottom-up order (a node
    always comes after every one of its own children -- directly a valid
    evaluation order, no reversal needed).

    A plain pre-order dedup (keep each node's first occurrence, then
    reverse) looks equivalent but isn't: when the same subformula recurs
    at different depths -- e.g. `Until(p, Next(p))`, or the `Const(1)`
    that `F`/`G`'s desugaring injects at every nesting level -- its first
    (shallowest) occurrence can land in a position that precedes a parent
    it is also a *child* of, so reversing produces an order where that
    parent is evaluated before its own child. Recursing into children
    first, and only recording a node once every child is already
    recorded, is correct regardless of how a subformula is shared.
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


def free_nodes(formula: Formula) -> tuple[Formula, ...]:
    """The subformulas of `formula` whose value a consistent state assigns
    freely: every `Prop`, `Next`, and `Until` node in `cl(formula)`.
    `Const`/`Apply` nodes are never free -- their value is forced by
    consistency alone."""
    return tuple(
        node for node in closure(formula) if isinstance(node, (Prop, Next, Until))
    )


def value_sets(
    formula: Formula,
    value_domains: Mapping[str, frozenset[Value] | set[Value]],
    registry: FunctionRegistry,
) -> dict[Formula, frozenset[Value]]:
    """`V(psi)` for every subformula `psi` of `formula`, generalized to
    weighted atoms: a `Prop`'s value set is given, not fixed to `{0,1}`.

    Args:
        value_domains: `V(p)` for every `Prop` name `p` in `formula`.

    Raises:
        KeyError: if some `Prop` in `formula` is missing from `value_domains`.
    """
    result: dict[Formula, frozenset[Value]] = {}
    for node in closure(formula):
        if isinstance(node, Const):
            result[node] = frozenset({node.value})
        elif isinstance(node, Prop):
            if node.name not in value_domains:
                raise KeyError(f"no value domain given for proposition {node.name!r}")
            result[node] = frozenset(value_domains[node.name])
        elif isinstance(node, Apply):
            result[node] = frozenset(
                registry.image(node.function, [result[arg] for arg in node.args])
            )
        elif isinstance(node, Next):
            result[node] = result[node.operand]
        elif isinstance(node, Until):
            result[node] = result[node.left] | result[node.right]
    return result
