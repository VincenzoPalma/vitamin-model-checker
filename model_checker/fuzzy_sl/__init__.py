"""Quantitative models and syntax for SL1G[F] (ATL*[F] included, as its
non-alternating fragment).

Public names are re-exported here.
"""

from __future__ import annotations

from .blocks import to_block_spec
from .formula import (
    Apply,
    Coalition,
    Const,
    Formula,
    Next,
    Prop,
    Quantifier,
    Quantify,
    Until,
    constants,
    is_state_formula,
    validate,
    value_set,
    value_set_labelings,
)
from .model import WCGS
from .model_adapter import adapt
from .parser import ParseError, parse
from .strategy_export import export_strategy
from .values import (
    FunctionRegistry,
    QualityFunction,
    Value,
    standard_registry,
    to_value,
)
from .verifier import Witness, check, holds, sat, value_at, witness

__all__ = [
    "WCGS",
    "Apply",
    "Coalition",
    "Const",
    "Formula",
    "FunctionRegistry",
    "Next",
    "ParseError",
    "Prop",
    "QualityFunction",
    "Quantifier",
    "Quantify",
    "Until",
    "Value",
    "Witness",
    "adapt",
    "check",
    "constants",
    "export_strategy",
    "holds",
    "is_state_formula",
    "parse",
    "sat",
    "standard_registry",
    "to_block_spec",
    "to_value",
    "validate",
    "value_at",
    "value_set",
    "value_set_labelings",
    "witness",
]
