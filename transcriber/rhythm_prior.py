"""Prior over within-beat rhythm patterns, learned from real scores.

A *pattern* is the sorted set of onset positions inside one metric beat, as
fractions of the beat: (0,) is a quarter in 4/4, (0, 1/2) two eighths,
(0, 1/3, 2/3) a triplet, (0, 3/4) a dotted-eighth + sixteenth, and so on.
`scripts/train_rhythm_prior.py` counts these over PDMX (public-domain
MuseScore scores); this module turns the counts into log-probabilities.

If the trained model is missing, a hand-written fallback prior is used that
encodes the same intuition: simple subdivisions are common, 1/32-beat
positions and odd tuplets are rare.
"""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction
from functools import lru_cache
from math import lcm
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "rhythm_prior.json"

# Positions we can notate inside a beat: duple down to 1/16 of a beat, triplet
# and sextuplet subdivisions, and quintuplets.
ALLOWED_DENOMINATORS = {1, 2, 3, 4, 5, 6, 8, 12, 16}
FAMILIES = ("plain", "duple", "triplet", "quint", "other")


def pattern_key(positions) -> str:
    return ",".join(str(Fraction(p)) for p in sorted(positions))


def parse_key(key: str) -> tuple[Fraction, ...]:
    return tuple(Fraction(p) for p in key.split(",")) if key else ()


def family(positions) -> str:
    d = 1
    for p in positions:
        d = lcm(d, Fraction(p).denominator)
    if d == 1:
        return "plain"
    if d & (d - 1) == 0:
        return "duple"
    if d % 5 == 0:
        return "quint"
    if d % 3 == 0 and (d // 3) & (d // 3 - 1) == 0:
        return "triplet"
    return "other"


def max_denominator(positions) -> int:
    d = 1
    for p in positions:
        d = lcm(d, Fraction(p).denominator)
    return d


class RhythmPrior:
    def __init__(self, patterns: dict[str, float], transitions: dict[str, float],
                 source: str):
        total = sum(patterns.values())
        self.source = source
        # Factorised as P(family) * P(pattern | family): the family transition
        # model below then says how likely a triplet beat is after a duple one,
        # without counting the rarity of triplets twice.
        fam_total: dict[str, float] = {}
        for k, c in patterns.items():
            f = family(parse_key(k))
            fam_total[f] = fam_total.get(f, 0) + c
        self.family_logp = {f: math.log(c / total) for f, c in fam_total.items()}
        self.logp = {parse_key(k): math.log(c / fam_total[family(parse_key(k))])
                     for k, c in patterns.items()}
        # Family transitions with add-one smoothing.
        self.trans: dict[tuple[str, str], float] = {}
        for a in FAMILIES:
            row = {b: transitions.get(f"{a}>{b}", 0) + 1 for b in FAMILIES}
            z = sum(row.values())
            for b, c in row.items():
                self.trans[(a, b)] = math.log(c / z)
        self.by_size: dict[int, list[tuple[tuple[Fraction, ...], float, str]]] = {}
        for pat, lp in self.logp.items():
            self.by_size.setdefault(len(pat), []).append((pat, lp, family(pat)))

    def candidates(self, n: int, allow) -> list[tuple[tuple[Fraction, ...], float, str]]:
        """Patterns with exactly `n` onsets that pass the `allow(pattern)` filter."""
        return [c for c in self.by_size.get(n, []) if allow(c[0])]

    def transition(self, prev: str | None, cur: str) -> float:
        """log P(family | previous family), or the marginal at the start."""
        if prev is None:
            return self.family_logp.get(cur, -10.0)
        return self.trans[(prev, cur)]


def _fallback_counts() -> tuple[dict[str, float], dict[str, float]]:
    """Synthetic prior: each pattern weighted by how simple its grid is."""
    grid_weight = {1: 1.0, 2: 0.5, 4: 0.12, 3: 0.06, 8: 0.01, 6: 0.008, 5: 0.002}
    counts: dict[str, float] = {}
    for d, w in grid_weight.items():
        points = [Fraction(i, d) for i in range(d)]
        for r in range(1, min(len(points), 6) + 1):
            for combo in itertools.combinations(points, r):
                if max_denominator(combo) != d and d != 1:
                    continue  # counted under its simpler grid
                # Every extra onset off the main grid costs a little more.
                counts[pattern_key(combo)] = counts.get(pattern_key(combo), 0) + \
                    1000 * w * (0.6 ** (r - 1))
    trans = {f"{a}>{b}": (50 if a == b else 5) for a in FAMILIES for b in FAMILIES}
    return counts, trans


@lru_cache(maxsize=1)
def load_prior() -> RhythmPrior:
    if MODEL_PATH.exists():
        data = json.loads(MODEL_PATH.read_text())
        patterns = data["patterns"]
        # Make sure the single-onset-on-the-beat case always exists.
        patterns.setdefault("0", 1)
        return RhythmPrior(patterns, data.get("family_transitions", {}),
                           f"trained on {data.get('files', '?')} PDMX scores")
    counts, trans = _fallback_counts()
    return RhythmPrior(counts, trans, "built-in fallback (no trained model found)")
