"""ATL*[F] parser entry point for `vitamin.parsers`: SL1G_F's parser without
quantifiers, so a formula's strategic parts are coalitions `<<...>>` only."""

from model_checker.parsers.formulas.SL1G_F.parser import SL1GFParser


class ATLStarFParser(SL1GFParser):
    """`parse` returns a `fuzzy_sl` formula, or None with the reason in `errors`."""

    allow_quantifiers = False
