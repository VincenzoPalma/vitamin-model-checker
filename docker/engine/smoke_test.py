"""Checks that the installed engine runs, including the logics that need Spot.

Expected values are worked out by hand. On the fuzzy model agent 1 has no choice
and agent 2 keeps the play in s1, where p is 0.2, so `F p` is worth 0.2 at s0.
"""

import platform
import sys

import spot

from model_checker.algorithms.explicit.ATL_STAR import ATL_STAR
from model_checker.algorithms.explicit.ATL_STAR_F import ATL_STAR_F
from model_checker.algorithms.explicit.ATLF import ATLF
from model_checker.algorithms.explicit.SL1G_F import SL1G_F

FUZZY = "model_checker/tests/fixtures/fuzzyCGS/ATLF/atlf_2agents_4states_fuzzy.txt"
NESTED = "examples/ATL_STAR/nested_coalition.txt"

CASES = [
    ("ATLF", ATLF.model_checking, "<1>F p", FUZZY, "Initial state s0: 0.2"),
    (
        "ATL_STAR_F",
        ATL_STAR_F.model_checking,
        "<<1>> F p",
        FUZZY,
        "Initial state s0: 0.2",
    ),
    (
        "SL1G_F",
        SL1G_F.model_checking,
        "E x A y: (x, 1)(y, 2) F p",
        FUZZY,
        "Initial state s0: 0.2",
    ),
    (
        "ATL_STAR",
        ATL_STAR.model_checking,
        "<<1>> F (<<2>> G goal)",
        NESTED,
        "Initial state s0: True",
    ),
    (
        "ATL_STAR synthesis",
        ATL_STAR.synthesize_strategy,
        "<<1>> F goal",
        NESTED,
        "State s0: True",
    ),
    (
        "SL1G_F synthesis",
        SL1G_F.synthesize_strategy,
        "E x A y: (x, 1)(y, 2) F p",
        FUZZY,
        "State s0: 0.2",
    ),
]


def main() -> int:
    print(
        f"{platform.machine()} | Python {platform.python_version()} | Spot {spot.version()}"
    )
    failed = 0
    for name, check, formula, model, expected in CASES:
        result = check(formula, model)
        ok = "error" not in result and result.get("initial_state") == expected
        if "synthesis" in name:
            ok = ok and bool(result.get("strategy", {}).get("moves"))
        failed += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {name}: {result.get('initial_state') or result.get('error')}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
