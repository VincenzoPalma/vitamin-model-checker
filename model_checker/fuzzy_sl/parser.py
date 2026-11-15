"""Parses SL1G[F] formula text into the AST of formula.py.

The syntax follows VITAMIN's shared formula conventions (its ATL* and NatSL
parsers), so the same text means the same thing in both tools.

Grammar, lowest to highest precedence:

    formula     := implication
    implication := disjunction ('->' implication)?      right-assoc
    disjunction := conjunction ('||' conjunction)*
    conjunction := until ('&&' until)*
    until       := unary ('U' unary)?                   no chaining, see below
    unary       := ('!' | 'X' | 'F' | 'G') unary
                 | prefix until
                 | atom
    prefix      := '<<' agents '>>' | '[[' agents ']]'
                 | (('E' | 'A') NAME)+ ':' ('(' NAME ',' agent ')'){n}
                                                    -- n = number of quantifiers just parsed
    atom        := NUMBER | 'true' | 'false' | '(' formula ')'
                 | NAME '(' (formula (',' formula)*)? ')'
                 | NAME
    agents      := (agent (',' agent)*)?

- `!`, `&&`, `||` and `->` are applications of the registered functions `not`,
  `and`, `or` and `implies`; each also has a lowercase word form (`not`, `and`,
  `or`, `implies`). Those words are connectives, never function names: the n-ary
  forms are `min(...)` and `max(...)`. A single `&` or `|` is an error.
- `X`, `F`, `G`, `U`, `E`, `A` are uppercase only and have the lowercase word
  forms `next`, `eventually`, `globally`/`always`, `until`, `exist`, `forall`.
- A name (proposition, function, agent or variable) starts with a lowercase
  letter and is no reserved word in any letter case, so `Goal`, `True` and
  `Until` are errors. `Xp` is one name: write `X p`.
- A prefix scopes over an Until: `<<1>> p U q` is `<<1>> (p U q)`. A connective
  right after a prefix is ambiguous and an error: write `(<<1>> F p) && q` or
  `<<1>> (F p && q)`. So is a second `U`, or a `U` next to a connective.
- `E x A y: (x, 1)(y, 2) psi` is SL1G[F]'s quantifier prefix (see `Quantify`),
  written as VITAMIN's NatSL writes it, without NatSL's `{k}` bounds. Every
  quantified variable is bound to exactly one agent after the colon.

`F`, `G` and `[[A]]` are sugar, desugared here:

    F psi      ->  1 U psi
    G psi      ->  not(1 U not(psi))
    [[A]] psi  ->  not(<<A>> not(psi))

An empty coalition `<<>>` is allowed and quantifies over all plays. Agent ids
are integers or names, and are not checked here: which agents exist, or whether
a binding names one twice, is either checked by `Quantify` itself or is a
question about a model, not about a formula.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from .formula import (
    AgentId,
    Apply,
    Coalition,
    Const,
    Formula,
    Next,
    Prop,
    Quantifier,
    Quantify,
    Until,
    Var,
    is_state_formula,
)
from .values import ONE, ZERO, FunctionRegistry, standard_registry, to_value


class ParseError(ValueError):
    """Raised on malformed formula text; `position` is 0-based."""

    def __init__(self, message: str, position: int):
        super().__init__(f"{message} (position {position})")
        self.position = position


_SINGLE_LETTER_KEYWORDS = {
    "X": "NEXT",
    "F": "EVENTUALLY",
    "G": "GLOBALLY",
    "U": "UNTIL",
    "E": "EXIST",
    "A": "FORALL",
}
_WORD_KEYWORDS = {
    "not": "NOT",
    "and": "AND",
    "or": "OR",
    "implies": "IMPLIES",
    "next": "NEXT",
    "eventually": "EVENTUALLY",
    "globally": "GLOBALLY",
    "always": "GLOBALLY",
    "until": "UNTIL",
    "exist": "EXIST",
    "forall": "FORALL",
    "true": "TRUE",
    "false": "FALSE",
}
# VITAMIN reserves these too, in every logic, though nothing here reads them.
_RESERVED_WORDS = frozenset(_WORD_KEYWORDS) | {"release", "weak", "with", "is"}
_NAME_RE = re.compile(r"[a-z][a-zA-Z0-9_]*")

_CONNECTIVES = frozenset({"AND", "OR", "IMPLIES"})
_QUANTIFIERS = frozenset({"EXIST", "FORALL"})
# What a connective word reads as when written like a function call.
_CALL_HINTS = {"AND": "min(...)", "OR": "max(...)", "IMPLIES": "x -> y"}

_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<COALITION_OPEN><<)
    | (?P<COALITION_CLOSE>>>)
    | (?P<DUAL_OPEN>\[\[)
    | (?P<DUAL_CLOSE>\]\])
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<COMMA>,)
    | (?P<COLON>:)
    | (?P<IMPLIES>->)
    | (?P<AND>&&)
    | (?P<OR>\|\|)
    | (?P<NOT>!)
    | (?P<NUMBER>\d+/\d+|\d+\.\d+|\d+)
    | (?P<IDENT>[a-zA-Z_][a-zA-Z0-9_]*)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


def _name_error(name: str) -> str | None:
    """Why `name` can't be a name, or None if it can."""
    if name.lower() in _RESERVED_WORDS:
        return f"{name!r} clashes with a reserved keyword"
    if not _NAME_RE.fullmatch(name):
        return f"{name!r} is not a valid name: a name starts with a lowercase letter, then letters, digits or _"
    return None


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            char = text[pos]
            hint = {
                "&": ": write && for conjunction",
                "|": ": write || for disjunction",
            }.get(char, "")
            raise ParseError(f"unexpected character {char!r}{hint}", pos)
        kind = cast(str, match.lastgroup)  # every alternative is a named group
        value = match.group()
        if kind != "WS":
            if kind == "IDENT":
                kind = (
                    _SINGLE_LETTER_KEYWORDS.get(value)
                    or _WORD_KEYWORDS.get(value)
                    or "IDENT"
                )
                if kind == "IDENT" and (error := _name_error(value)) is not None:
                    raise ParseError(error, pos)
            tokens.append(_Token(kind, value, pos))
        pos = match.end()
    tokens.append(_Token("EOF", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token], registry: FunctionRegistry):
        self._tokens = tokens
        self._index = 0
        self._registry = registry

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _expect(self, kind: str) -> _Token:
        token = self._peek()
        if token.kind != kind:
            raise ParseError(f"expected {kind}, found {token.kind}", token.position)
        return self._advance()

    def _apply(self, name: str, args: tuple[Formula, ...], position: int) -> Apply:
        """Build an `Apply`, checking `name` against F at parse time."""
        if name not in self._registry:
            raise ParseError(
                f"{name!r} is not in F; registered: {sorted(self._registry.names())}",
                position,
            )
        try:
            self._registry.require_arity(name, len(args))
        except ValueError as exc:
            raise ParseError(str(exc), position) from exc
        return Apply(name, args)

    def parse_formula(self) -> Formula:
        start = self._peek().position
        formula = self._implication()
        self._expect("EOF")
        if not is_state_formula(formula):
            raise ParseError(
                "a formula must have a value at a state, but this one needs a path",
                start,
            )
        return formula

    def _implication(self) -> Formula:
        left = self._disjunction()
        if self._peek().kind == "IMPLIES":
            token = self._advance()
            return self._apply("implies", (left, self._implication()), token.position)
        return left

    def _disjunction(self) -> Formula:
        left = self._conjunction()
        while self._peek().kind == "OR":
            token = self._advance()
            left = self._apply("or", (left, self._conjunction()), token.position)
        return left

    def _conjunction(self) -> Formula:
        left = self._until()
        while self._peek().kind == "AND":
            token = self._advance()
            left = self._apply("and", (left, self._until()), token.position)
        return left

    def _until(self) -> Formula:
        start = self._index
        left = self._unary()
        if self._peek().kind != "UNTIL":
            return left
        until = self._advance()
        right = self._unary()
        after = self._peek()
        if after.kind == "UNTIL":
            raise ParseError(
                "U does not chain: write (x U y) U z or x U (y U z)", after.position
            )
        # No precedence between U and the connectives, as in VITAMIN.
        before = self._tokens[start - 1] if start else None
        if before is not None and before.kind in _CONNECTIVES:
            c = before.value
            raise ParseError(
                f"'x {c} y U z' is ambiguous: write (x {c} y) U z or x {c} (y U z)",
                until.position,
            )
        if after.kind in _CONNECTIVES:
            c = after.value
            raise ParseError(
                f"'x U y {c} z' is ambiguous: write (x U y) {c} z or x U (y {c} z)",
                until.position,
            )
        return Until(left, right)

    def _unary(self) -> Formula:
        token = self._peek()
        match token.kind:
            case "NOT":
                self._advance()
                return self._apply("not", (self._unary(),), token.position)
            case "NEXT":
                self._advance()
                return Next(self._unary())
            case "EVENTUALLY":
                self._advance()
                return Until(Const(ONE), self._unary())
            case "GLOBALLY":
                self._advance()
                body = self._apply("not", (self._unary(),), token.position)
                return self._apply("not", (Until(Const(ONE), body),), token.position)
            case "COALITION_OPEN":
                self._advance()
                agents = self._agents("COALITION_CLOSE")
                return Coalition(
                    agents,
                    self._prefix_body(token, self._coalition_text("<<", agents, ">>")),
                )
            case "DUAL_OPEN":
                self._advance()
                agents = self._agents("DUAL_CLOSE")
                body = self._prefix_body(
                    token, self._coalition_text("[[", agents, "]]")
                )
                return self._apply(
                    "not",
                    (Coalition(agents, self._apply("not", (body,), token.position)),),
                    token.position,
                )
            case "EXIST" | "FORALL":
                return self._quantify()
        return self._atom()

    @staticmethod
    def _coalition_text(opening: str, agents: frozenset[AgentId], closing: str) -> str:
        return f"{opening}{','.join(sorted(map(str, agents)))}{closing}"

    def _prefix_body(self, opening: _Token, prefix: str) -> Formula:
        """The Until a prefix scopes over, rejecting a connective right after it:
        `<<1>> x && y` reads as `(<<1>> x) && y` or `<<1>> (x && y)`."""
        body = self._until()
        after = self._peek()
        if after.kind in _CONNECTIVES:
            c = after.value
            raise ParseError(
                f"'{prefix} ... {c} ...' is ambiguous: write {prefix} (... {c} ...) for {prefix} over both "
                f"sides, or ({prefix} ...) {c} ... for {prefix} over the left side only",
                opening.position,
            )
        return body

    def _name_token(self, what: str = "a name") -> AgentId | Var:
        """An agent id or a quantified variable -- lexically identical:
        `NUMBER` (as an int) or `IDENT` (as-is)."""
        token = self._peek()
        if token.kind == "NUMBER" and token.value.isdigit():
            self._advance()
            return int(token.value)
        if token.kind == "IDENT":
            self._advance()
            return token.value
        raise ParseError(f"expected {what}, found {token.kind}", token.position)

    def _agents(self, closing: str) -> frozenset[AgentId]:
        agents: dict[AgentId, None] = {}

        def add_one() -> None:
            token = self._peek()
            agent = self._name_token("an agent")
            if agent in agents:
                raise ParseError(
                    f"duplicate agent {agent!r} in coalition", token.position
                )
            agents[agent] = None

        # Mirrors `_arguments`'s shape: a comma must be followed by another
        # agent, not the closing bracket -- no trailing comma, same as a
        # function call's argument list.
        if self._peek().kind != closing:
            add_one()
            while self._peek().kind == "COMMA":
                self._advance()
                add_one()
        self._expect(closing)
        return frozenset(agents)

    def _quantify(self) -> Formula:
        opening = self._peek()
        prefix: list[tuple[Quantifier, Var]] = []
        while self._peek().kind in _QUANTIFIERS:
            token = self._advance()
            quantifier = (
                Quantifier.EXISTS if token.kind == "EXIST" else Quantifier.FORALL
            )
            prefix.append((quantifier, self._name_token("a variable")))
        self._expect("COLON")

        bindings = tuple(self._binding() for _ in range(len(prefix)))
        shown = (
            " ".join(f"{'E' if q is Quantifier.EXISTS else 'A'} {v}" for q, v in prefix)
            + ":"
        )
        body = self._prefix_body(opening, shown)

        try:
            return Quantify(tuple(prefix), bindings, body)
        except ValueError as exc:
            raise ParseError(str(exc), opening.position) from exc

    def _binding(self) -> tuple[AgentId, Var]:
        """`(variable, agent)`, as NatSL writes it; returned agent first, as
        `Quantify.bindings` stores it."""
        self._expect("LPAREN")
        variable = self._name_token("a variable")
        self._expect("COMMA")
        agent = self._name_token("an agent")
        self._expect("RPAREN")
        return (agent, variable)

    def _atom(self) -> Formula:
        token = self._peek()
        match token.kind:
            case "TRUE":
                self._advance()
                return Const(ONE)
            case "FALSE":
                self._advance()
                return Const(ZERO)
            case "NUMBER":
                self._advance()
                try:
                    return Const(to_value(token.value))
                except ValueError as exc:
                    raise ParseError(str(exc), token.position) from exc
            case "IDENT":
                self._advance()
                if self._peek().kind == "LPAREN":
                    return self._apply(token.value, self._arguments(), token.position)
                return Prop(token.value)
            case "LPAREN":
                self._advance()
                inner = self._implication()
                self._expect("RPAREN")
                return inner
        if token.kind in _CALL_HINTS and self._tokens[self._index + 1].kind == "LPAREN":
            raise ParseError(
                f"{token.value!r} is a connective, not a function: write {_CALL_HINTS[token.kind]}",
                token.position,
            )
        raise ParseError(f"unexpected token {token.kind}", token.position)

    def _arguments(self) -> tuple[Formula, ...]:
        self._expect("LPAREN")
        args: list[Formula] = []
        if self._peek().kind != "RPAREN":
            args.append(self._implication())
            while self._peek().kind == "COMMA":
                self._advance()
                args.append(self._implication())
        self._expect("RPAREN")
        return tuple(args)


def parse(text: str, registry: FunctionRegistry | None = None) -> Formula:
    """Parse formula text into a `Formula`.

    Args:
        text: the formula text.
        registry: F, against which every function named is checked for
            existence and arity. Defaults to `standard_registry()`.

    Raises:
        ParseError: on malformed or ambiguous text, an unknown or misapplied
            function, a constant outside [0,1], or a formula that needs a
            path to have a value.
    """
    tokens = _tokenize(text)
    return _Parser(
        tokens, standard_registry() if registry is None else registry
    ).parse_formula()
