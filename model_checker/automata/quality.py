"""The generic function registry LTL[F] needs: `ltlf.Apply` names a
function by a plain string, and this is what resolves that name to a
callable + arity (`F: [0,1]^m -> [0,1]`).

Generic and logic-agnostic like the rest of this package -- unlike a client
repo's own registry (e.g. `fuzzy_sl.values.FunctionRegistry`), this one
defines no standard connectives (`and`/`or`/`not`/...): which functions
exist is a question about a particular logic, not about LTL[F] itself. A
caller with its own registry-like object translates it into a
`FunctionRegistry` here (or builds one directly) before calling anything
in this package that needs one.
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

    Accepts ints, `Fraction`s, `Decimal`s and strings like `"1/2"` or
    `"0.5"`. Floats work but give the exact binary double.

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
    """An open collection of named quality functions -- one instantiation of F."""

    def __init__(self) -> None:
        self._functions: dict[str, QualityFunction] = {}

    def register(self, name: str, fn: Callable[..., object], arity: int | None) -> None:
        """Add a function under `name`.

        Raises:
            ValueError: if `name` is taken or `arity` is negative.
        """
        if name in self._functions:
            raise ValueError(f"function {name!r} is already registered")
        if arity is not None and arity < 0:
            raise ValueError(f"arity of {name!r} must be non-negative, got {arity}")
        self._functions[name] = QualityFunction(name, arity, fn)

    def names(self) -> frozenset[str]:
        return frozenset(self._functions)

    def __contains__(self, name: object) -> bool:
        return name in self._functions

    def __getitem__(self, name: str) -> QualityFunction:
        try:
            return self._functions[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not registered; registered: {sorted(self._functions)}"
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
        """`{f(v1..vm) | vi in value_sets[i]}` -- one evaluation per point of
        the cartesian product."""
        function = self[name]
        materialized = [[to_value(value) for value in values] for values in value_sets]
        self.require_arity(name, len(materialized))
        return {
            self._call(function, combination) for combination in product(*materialized)
        }

    def require_arity(self, name: str, count: int) -> None:
        """Raise unless `name` accepts `count` arguments.

        Raises:
            KeyError: if `name` is not registered.
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
