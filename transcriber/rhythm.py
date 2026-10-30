"""Quantization: from performed note times to notated rhythms.

Times are first mapped onto the (variable-tempo) beat grid, so rubato and
tempo changes don't move notes off the beat. Within each beat, the onsets are
then explained by the most probable *written* rhythm pattern:

    cost(pattern) = timing error (Gaussian, ~35 ms)  -  log P(pattern)
                    -  log P(pattern's family | previous beat's family)

P(pattern) is the rhythm prior learned from PDMX scores (see
rhythm_prior.py), so a clean triplet or quintuplet beats an odd 32nd/64th
figure unless the timing really demands it. The family term (duple /
triplet / quintuplet) makes runs of triplets hang together. A Viterbi pass
over beats picks the best sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import librosa

from .beats import BeatGrid
from .notes import Note
from .rhythm_prior import RhythmPrior, family, load_prior

TIMING_SIGMA_S = 0.035   # performance + detection jitter
MIN_SIGMA_BEATS = 0.035
CLUSTER_S = 0.04         # onsets this close count as one (chords, rolled chords)
EARLY_SLOP = 0.12        # an onset this early (in beats) still belongs to the next beat
# Extra cost (in nats) for a tuplet beat on top of the learned prior. The
# PDMX corpus contains plenty of virtuoso music; for transcription we want a
# tuplet only when the timing clearly calls for it.
TUPLET_COST = {"triplet": 1.0, "quint": 6.0, "other": 6.0}
FINER_GRID_COST = 1.5    # using 32nds because a beat holds more notes than 16ths allow


@dataclass
class QuantizedNote:
    offset: Fraction    # in beats from beat 0 of the grid
    duration: Fraction  # in beats
    pitch: int
    amplitude: float
    graces: list[int] = field(default_factory=list)
    trill: bool = False
    raw_end: float = 0.0  # detected release, in beats (before any snapping)


@dataclass
class RhythmOptions:
    subdivision: int = 4      # finest duple subdivision per beat (4 = 16ths in 4/4)
    tuplets: bool = True      # allow triplets / sextuplets
    quintuplets: bool = False  # rare in practice; off unless asked for


def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    """Global tempo in BPM, folded into a readable 60-160 range."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    bpm = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                      start_bpm=100)[0])
    if bpm <= 0:
        return 100.0
    while bpm < 60:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return bpm


def fit_phase(onsets: np.ndarray, bpm: float) -> float:
    """Time of a beat line that best lines up with the onsets (constant tempo)."""
    beat_s = 60.0 / bpm
    z = np.mean(np.exp(2j * np.pi * onsets / beat_s))
    phase = float(np.angle(z) / (2 * np.pi) * beat_s)
    return phase + np.floor((onsets.min() - phase) / beat_s + 0.125) * beat_s


def _allow(opts: RhythmOptions, subdivision: int | None = None):
    sub = subdivision or opts.subdivision

    def ok(pattern) -> bool:
        for p in pattern:
            d = Fraction(p).denominator
            if d & (d - 1) == 0:          # power of two
                if d > sub:
                    return False
            elif d % 5 == 0:
                if not (opts.tuplets and opts.quintuplets):
                    return False
            elif not opts.tuplets:
                return False
            elif d == 12 and sub < 8:
                return False
        return True
    return ok


def _end_grid(fam: str | None, opts: RhythmOptions) -> Fraction:
    if fam == "triplet":
        return Fraction(1, 3)
    if fam == "quint":
        return Fraction(1, 5)
    return Fraction(1, min(opts.subdivision, 4))


def _cluster(notes: list[Note], grid: BeatGrid) -> list[tuple[float, list[int]]]:
    """Group near-simultaneous onsets; returns (beat position, note indices).

    The window shrinks at fast tempi so consecutive notes of a 32nd-note run
    are never mistaken for one chord.
    """
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    clusters: list[list[int]] = []
    for i in order:
        t = notes[i].start
        window = min(CLUSTER_S, 0.35 * grid.beat_seconds(int(grid.to_beats(t))) / 8)
        if clusters and t - notes[clusters[-1][0]].start <= window:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    out = []
    for c in clusters:
        t = float(np.mean([notes[i].start for i in c]))
        out.append((float(grid.to_beats(t)), c))
    return out


RARE_LOGP = -6.0  # log P(pattern | duple) for plain runs the corpus didn't list


def _runs_on_grid(n: int, sub: int) -> list[tuple[Fraction, ...]]:
    """Every way to place n onsets on a 1/sub grid (only for small grids)."""
    if n > sub or sub > 8:
        return []
    from itertools import combinations
    return [tuple(Fraction(k, sub) for k in c) for c in combinations(range(sub), n)]


def _fallback_pattern(rs: list[float]) -> tuple[Fraction, ...]:
    """When no learned pattern has this many onsets: keep every onset, in order.

    Uses the coarsest duple grid with room for all of them and nudges
    collisions forward, so a fast run is never squashed into a chord.
    """
    n = len(rs)
    step = Fraction(1, 4)
    while 1 / step < n:
        step /= 2
    out, prev = [], -1
    for r in rs:
        k = max(int(round(r / float(step))), prev + 1)
        out.append(k)
        prev = k
    # If nudging ran past the beat, spread evenly instead.
    if out[-1] * step >= 1:
        out = list(range(n))
        step = Fraction(1, n) if n & (n - 1) == 0 else step
    return tuple(k * step for k in out)


def quantize_part(notes: list[Note], grid: BeatGrid, opts: RhythmOptions | None = None,
                  prior: RhythmPrior | None = None) -> tuple[list[QuantizedNote], dict[int, str]]:
    """Quantize note onsets. Returns notes and each beat's rhythm family.

    Releases are only snapped provisionally here; where a note really stops
    depends on the next note in its own voice, which voices.py decides.
    """
    opts = opts or RhythmOptions()
    prior = prior or load_prior()
    if not notes:
        return [], {}
    allow = _allow(opts)

    clusters = _cluster(notes, grid)
    xs = [x for x, _ in clusters]
    by_beat: dict[int, list[int]] = {}
    for ci, x in enumerate(xs):
        # An onset slightly before a beat belongs to that beat, but in a fast
        # run the last 32nd of a beat is itself only 1/8 beat early, so the
        # tolerance shrinks with the spacing of neighbouring onsets.
        gaps = [abs(x - xs[j]) for j in (ci - 1, ci + 1) if 0 <= j < len(xs)]
        slop = min([EARLY_SLOP] + [0.4 * g for g in gaps])
        by_beat.setdefault(int(np.floor(x + slop)), []).append(ci)
    beats = sorted(by_beat)

    # Viterbi over beats; state = rhythm family of the last beat with onsets.
    # best[fam] = (cost, backpointer chain index)
    trellis: list[dict[str, tuple[float, str | None, tuple[Fraction, ...]]]] = []
    prev: dict[str | None, float] = {None: 0.0}
    for b in beats:
        rs = [clusters[ci][0] - b for ci in by_beat[b]]
        sigma = max(TIMING_SIGMA_S / grid.beat_seconds(b), MIN_SIGMA_BEATS)
        cands = [(c, 0.0) for c in prior.candidates(len(rs), allow)]
        if not cands or len(rs) >= 5:
            # More notes in this beat than the grid holds (a fast run):
            # allow the next finer subdivision rather than squashing them,
            # and always offer plain runs on that grid, since the corpus
            # rarely has e.g. exactly seven 32nds in a beat.
            sub = min(opts.subdivision * 2, 16) if len(rs) > opts.subdivision \
                else opts.subdivision
            finer = _allow(opts, sub)
            cands += [(c, FINER_GRID_COST) for c in prior.candidates(len(rs), finer)
                      if c not in [x for x, _ in cands]]
            cands += [((pat, RARE_LOGP, "duple"), FINER_GRID_COST)
                      for pat in _runs_on_grid(len(rs), sub)]
        options: list[tuple[tuple[Fraction, ...], float, str]] = []
        for (pat, logp, fam), extra in cands:
            err = sum((r - float(p)) ** 2 for r, p in zip(rs, pat))
            cost = err / (2 * sigma ** 2) - logp + extra + TUPLET_COST.get(fam, 0.0)
            options.append((pat, cost, fam))
        if not options:
            pat = _fallback_pattern(rs)
            options.append((pat, 50.0, family(pat)))
        col: dict[str, tuple[float, str | None, tuple[Fraction, ...]]] = {}
        for pat, cost, fam in options:
            for pfam, pcost in prev.items():
                total = pcost + cost - prior.transition(pfam, fam)
                if fam not in col or total < col[fam][0]:
                    col[fam] = (total, pfam, pat)
        trellis.append(col)
        prev = {f: v[0] for f, v in col.items()}

    # Backtrack.
    chosen: dict[int, tuple[tuple[Fraction, ...], str]] = {}
    fam = min(trellis[-1], key=lambda f: trellis[-1][f][0])
    for k in range(len(beats) - 1, -1, -1):
        _, pfam, pat = trellis[k][fam]
        chosen[beats[k]] = (pat, fam)
        fam = pfam

    onset_of: dict[int, Fraction] = {}
    for b in beats:
        pat, _ = chosen[b]
        for ci, p in zip(by_beat[b], pat):
            for i in clusters[ci][1]:
                onset_of[i] = b + Fraction(p)
    families = {b: f for b, (_, f) in chosen.items()}

    def snap_end(x: float) -> Fraction:
        b = int(np.floor(x))
        g = _end_grid(families.get(b), opts)
        return b + Fraction(round((x - b) / float(g))) * g

    out: dict[tuple[Fraction, int], QuantizedNote] = {}
    for i, n in enumerate(notes):
        start = onset_of[i]
        raw_end = float(grid.to_beats(n.end))
        end = snap_end(raw_end)
        if end <= start:
            end = start + _end_grid(families.get(int(np.floor(start))), opts)
        key = (start, n.pitch)
        # Two detections collapsed onto the same grid point: keep the longer.
        if key not in out or out[key].duration < end - start:
            out[key] = QuantizedNote(start, end - start, n.pitch, n.amplitude,
                                     list(n.graces), n.trill, raw_end)
    qnotes = sorted(out.values(), key=lambda q: (q.offset, q.pitch))
    return trim_overlaps(qnotes), families


def is_compound(families: dict[int, str]) -> bool:
    """Is the music mostly in triplet subdivisions (so 6/8, 9/8, 12/8)?

    Needs triplets to dominate the subdivided beats *and* to be a sizeable
    share of all beats, so a few ornaments or fills don't flip the meter.
    """
    fams = list(families.values())
    active = [f for f in fams if f != "plain"]
    triplet = sum(f == "triplet" for f in active)
    return bool(active) and triplet > 0.6 * len(active) and triplet >= max(4, 0.3 * len(fams))


def trim_overlaps(qnotes: list[QuantizedNote]) -> list[QuantizedNote]:
    """A pitch can't sound twice at once: cut each note off at its next re-attack."""
    by_pitch: dict[int, list[QuantizedNote]] = {}
    for q in qnotes:
        by_pitch.setdefault(q.pitch, []).append(q)
    for seq in by_pitch.values():
        for a, b in zip(seq, seq[1:]):
            if a.offset + a.duration > b.offset:
                a.duration = b.offset - a.offset
    return qnotes
