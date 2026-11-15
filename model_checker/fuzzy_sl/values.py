"""Satisfaction values and the parameter F of ATL*[F].

A quality function is any `f: [0,1]^m -> [0,1]`; the logic puts no further
condition on F. F is therefore not a parameter here but an open registry, which
still needs each function's arity as well as its body: the set of values a
formula can take is built inductively, and the function case is an image over a
cartesian product of m value sets (see `FunctionRegistry.image`).

Values are exact rationals: float rounding would invent spurious values and
make threshold comparisons unreliable. A function whose true value is irrational
must return a rational approximation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import product

Value = Fraction

ZERO = Fraction(0)
ONE = Fraction(1)


def to_value(x: object) -> Value:
    """Convert to an exact `Value`, rejecting anything outside [0,1].

    Accepts ints, `Fraction`s, `Decimal`s and strings like `"1/2"` or `"0.5"`.
    Floats work but give the exact binary double, so `to_value(0.1)` is not
    `1/10`.

    Raises:
        ValueError: if the input is unreadable or falls outside [0,1].
    """
    try:
        value = Fraction(Decimal(x)) if isinstance(x, (Decimal, float)) else Fraction(x)  # type: ignore[arg-type]
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"cannot read {x!r} as a value") from exc
    if not ZERO <= value <= ONE:
        raise ValueError(f"value {value} is outside [0,1]")
    return value


@dataclass(frozen=True)
class QualityFunction:

    name: str
    arity: int | None
    fn: Callable[..., object]

    def accepts(self, count: int) -> bool:
        """Whether this function can be applied to `count` arguments."""
        return count >= 1 if self.arity is None else count == self.arity


class FunctionRegistry:
    """The set F, as an open collection of named quality functions."""

    def __init__(self) -> None:
        self._functions: dict[str, QualityFunction] = {}

    def register(self, name: str, fn: Callable[..., object], arity: int | None) -> None:
        """Add a function to F under `name`.

        Raises:
            ValueError: if `name` is taken or `arity` is negative.
        """
        if name in self._functions:
            raise ValueError(f"function {name!r} is already registered")
        if arity is not None and arity < 0:
            raise ValueError(f"arity of {name!r} must be non-negative, got {arity}")
        self._functions[name] = QualityFunction(name, arity, fn)

    def names(self) -> frozenset[str]:
        """The names currently in F."""
        return frozenset(self._functions)

    def __contains__(self, name: object) -> bool:
        return name in self._functions

    def __getitem__(self, name: str) -> QualityFunction:
        try:
            return self._functions[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not in F; registered: {sorted(self._functions)}"
            ) from None

    def apply(self, name: str, args: Sequence[object]) -> Value:
        """Evaluate `name` on `args`, checking arity, domain and codomain.

        Raises:
            ValueError: on a wrong argument count, or an argument or result
                outside [0,1].
        """
        self.require_arity(name, len(args))
        return self._call(self[name], [to_value(arg) for arg in args])

    def image(self, name: str, value_sets: Sequence[Iterable[Value]]) -> set[Value]:
        """`{f(v1..vm) | vi in value_sets[i]}`.

        The function case of the finiteness argument for satisfaction values,
        and the only case that creates values not already present. Costs one
        evaluation per point of the cartesian product, so the checks `apply`
        does per call are hoisted out of the loop here.
        """
        function = self[name]
        materialized = [[to_value(value) for value in values] for values in value_sets]
        self.require_arity(name, len(materialized))
        return {
            self._call(function, combination) for combination in product(*materialized)
        }

    def require_arity(self, name: str, count: int) -> None:
        """Raise unless `name` accepts `count` arguments.

        Raises:
            KeyError: if `name` is not in F.
            ValueError: on the wrong argument count.
        """
        function = self[name]
        if not function.accepts(count):
            expected = "at least 1" if function.arity is None else str(function.arity)
            raise ValueError(f"{name!r} takes {expected} argument(s), got {count}")

    @staticmethod
    def _call(function: QualityFunction, values: Sequence[Value]) -> Value:
        try:
            return to_value(function.fn(*values))
        except ValueError as exc:
            raise ValueError(
                f"{function.name!r} returned an invalid value: {exc}"
            ) from exc


def _clamp(value: Fraction | int) -> Fraction:
    return max(ZERO, min(ONE, Fraction(value)))


def standard_registry() -> FunctionRegistry:
    """A fresh registry with the usual fuzzy connectives

    Fresh per call, so registering a function never mutates anyone else's F.
    The three readings of the connectives coexisting without touching the logic
    is the point of parameterizing by F.
    """
    registry = FunctionRegistry()

    # Zadeh, the reading the logic's derived operators assume. The parser
    # reaches these four only through `!`, `&&`, `||` and `->`.
    registry.register("not", lambda x: ONE - x, 1)
    # Wrapped: bare `min` reads a single argument as an iterable.
    registry.register("and", lambda *xs: min(xs), None)
    registry.register("or", lambda *xs: max(xs), None)
    registry.register("implies", lambda x, y: max(ONE - x, y), 2)
    # The n-ary forms, callable by name.
    registry.register("min", lambda *xs: min(xs), None)
    registry.register("max", lambda *xs: max(xs), None)

    # Goedel-Dummett differs from Zadeh only in implication.
    registry.register("godel_implies", lambda x, y: ONE if x <= y else y, 2)

    registry.register("luk_and", lambda *xs: _clamp(sum(xs) - (len(xs) - 1)), None)
    registry.register("luk_or", lambda *xs: _clamp(sum(xs)), None)
    registry.register("luk_implies", lambda x, y: min(ONE, ONE - x + y), 2)

    # The `<=` test for Nash equilibria, the deviation gain for
    # epsilon-equilibria, and utilitarian social welfare.
    registry.register("leq", lambda x, y: ONE if x <= y else ZERO, 2)
    registry.register("diff", lambda x, y: max(ZERO, x - y), 2)
    registry.register("avg", lambda *xs: sum(xs) / len(xs), None)

    return registry
