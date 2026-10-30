"""Voice separation and note lengths: one part per musical line.

Notes are split into voices the way a musician hears them, by pitch and
voice leading rather than by timbre: a held bass line is one voice, the
moving melody above it another, accompaniment chords a third.

1. Chords. Notes struck together with (nearly) the same length form one
   chord event, unless they're more than an octave apart. A bass note plus
   a right-hand chord, or a held note under a moving one, stay separate.
2. Assignment. Moving through the music onset by onset, each event joins the
   running voice it continues best (Hungarian matching): small leap from the
   voice's last pitch and its usual register, no crossing a voice that's
   still sounding, a cost for cutting off a note that's still ringing, and a
   cost for starting a new voice.
3. Clean-up. Voices with only a handful of notes merge into their nearest
   neighbour; voices are ordered top to bottom.
4. Lengths. Within a voice, a note that sounds into (or nearly up to) the
   next note simply lasts until it. Otherwise it ends on the nearest simple
   value (quarter, half, dotted quarter...) and a rest follows. Chord tones
   share one length, so chords never turn into stacks of tied notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rhythm import QuantizedNote

CHORD_SPAN = 12        # notes further apart than this aren't one chord
NEW_VOICE_COST = 10.0  # in (leap-weighted) semitones
CROSSING_COST = 6.0
CUT_COST = 1.5         # per beat a still-ringing note would be cut short
FULL_COST = 40.0       # extra cost of a voice beyond the maximum

F = Fraction
# Written lengths (in beats) a note may end on when a rest follows, with a
# small preference order: plain values first, then dotted, then tuplets.
_SIMPLE = [(F(1), 0.0), (F(2), 0.0), (F(1, 2), 0.0), (F(4), 0.0), (F(3), 0.2),
           (F(3, 2), 0.2), (F(1, 4), 0.3), (F(3, 4), 0.3), (F(6), 0.4), (F(8), 0.2),
           (F(1, 3), 0.5), (F(2, 3), 0.5)]


@dataclass
class Event:
    offset: Fraction
    notes: list[QuantizedNote]

    @property
    def pitch(self) -> float:
        return float(np.mean([q.pitch for q in self.notes]))

    @property
    def duration(self) -> Fraction:
        return max(q.duration for q in self.notes)

    @property
    def raw_end(self) -> float:
        return max(q.raw_end for q in self.notes)


@dataclass
class Voice:
    events: list[Event] = field(default_factory=list)
    mean: float = 0.0

    @property
    def last(self) -> Event:
        return self.events[-1]

    def add(self, e: Event) -> None:
        n = len(self.events)
        w = 1 / (n + 1) if n < 6 else 0.15  # running register, settles after a few notes
        self.mean = e.pitch if n == 0 else (1 - w) * self.mean + w * e.pitch
        self.events.append(e)

    @property
    def notes(self) -> list[QuantizedNote]:
        return [q for e in self.events for q in e.notes]


def _events(qnotes: list[QuantizedNote]) -> list[list[Event]]:
    """Group notes into chord events, per onset time."""
    by_onset: dict[Fraction, list[QuantizedNote]] = {}
    for q in qnotes:
        by_onset.setdefault(q.offset, []).append(q)
    groups = []
    for onset in sorted(by_onset):
        notes = sorted(by_onset[onset], key=lambda q: q.duration)
        # Cluster by length: a held note and a short one struck together are
        # two different lines, not a chord.
        clusters: list[list[QuantizedNote]] = []
        for q in notes:
            if clusters and q.duration <= clusters[-1][0].duration * F(13, 10) + F(1, 4):
                clusters[-1].append(q)
            else:
                clusters.append([q])
        events = []
        for c in clusters:
            c.sort(key=lambda q: q.pitch)
            part = [c[0]]
            for q in c[1:]:
                if q.pitch - part[0].pitch > CHORD_SPAN:
                    events.append(Event(onset, part))
                    part = [q]
                else:
                    part.append(q)
            events.append(Event(onset, part))
        groups.append(sorted(events, key=lambda e: -e.pitch))
    return groups


def _leap(d: float) -> float:
    """Cost of a melodic leap: steps are cheap, and beyond a fifth each extra
    semitone costs little, because octave and fifth jumps are idiomatic
    (especially in bass lines)."""
    return min(d, 7.0) + 0.3 * max(d - 7.0, 0.0)


def _cost(e: Event, v: Voice, voices: list[Voice]) -> float:
    last = v.last
    idle = float(e.offset - last.offset)
    w = 0.7 if idle <= 4 else 0.4  # after a long rest, register matters more than the last note
    c = w * _leap(abs(e.pitch - last.pitch)) + (1 - w) * _leap(abs(e.pitch - v.mean))
    overhang = last.raw_end - float(e.offset)
    if overhang > 0:
        c += CUT_COST * min(overhang, 2.0)
    for u in voices:
        if u is v or not u.events:
            continue
        ul = u.last
        if ul.offset < e.offset < ul.offset + ul.duration:  # u still sounding
            if (v.mean > u.mean) != (e.pitch > ul.pitch):
                c += CROSSING_COST
    return c


def _merge_small(voices: list[Voice], max_voices: int, total: int) -> list[Voice]:
    min_events = max(3, int(0.04 * total))
    voices = [v for v in voices if v.events]
    while len(voices) > 1:
        small = min(voices, key=lambda v: len(v.events))
        if len(small.events) >= min_events and len(voices) <= max_voices:
            break
        voices.remove(small)
        target = min(voices, key=lambda v: abs(v.mean - small.mean))
        by_offset = {e.offset: e for e in target.events}
        for e in small.events:
            if e.offset in by_offset:  # same moment: becomes one chord
                by_offset[e.offset].notes.extend(e.notes)
            else:
                by_offset[e.offset] = e
        target.events = [by_offset[o] for o in sorted(by_offset)]
    return sorted(voices, key=lambda v: -v.mean)


def _span(v: Voice) -> tuple[Fraction, Fraction]:
    return v.events[0].offset, max(e.offset + e.duration for e in v.events)


def _join_successive(voices: list[Voice]) -> list[Voice]:
    """Join voices that take turns rather than sound together.

    If one line stops and another starts in a neighbouring register (a bass
    line that drops an octave, a melody that moves up after a rest), they
    are really one part.
    """
    changed = True
    while changed:
        changed = False
        for a in voices:
            for b in voices:
                if a is b or not a.events or not b.events:
                    continue
                a0, a1 = _span(a)
                b0, b1 = _span(b)
                if a1 <= b0 + F(1, 2) and abs(a.mean - b.mean) <= 12:
                    # b may start up to half a beat before a's last note ends
                    a.events = sorted(a.events + b.events, key=lambda e: e.offset)
                    a.mean = (a.mean * len(a.events) + b.mean * len(b.events)) / \
                        (len(a.events) + len(b.events))
                    b.events = []
                    changed = True
        voices = [v for v in voices if v.events]
    return voices


def separate_voices(qnotes: list[QuantizedNote], max_voices: int = 4) -> list[Voice]:
    groups = _events(qnotes)
    total = sum(len(g) for g in groups)
    voices: list[Voice] = []
    for group in groups:
        free = [v for v in voices]
        n_new = len(group)
        cost = np.zeros((len(group), len(free) + n_new))
        for i, e in enumerate(group):
            for j, v in enumerate(free):
                cost[i, j] = _cost(e, v, voices)
            for k in range(n_new):
                extra = FULL_COST if len(voices) + k >= max_voices else 0.0
                cost[i, len(free) + k] = NEW_VOICE_COST + extra
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if j < len(free):
                free[j].add(group[i])
            else:
                v = Voice()
                v.add(group[i])
                voices.append(v)
    return _merge_small(_join_successive(voices), max_voices, total)


def on_beat_grid(x: Fraction, families: dict[int, str]) -> bool:
    frac = x - int(x)
    if not frac:
        return True
    d = frac.denominator
    fam = families.get(int(x))
    if fam == "triplet":
        return d in (3, 6, 12)
    if fam == "quint":
        return d in (5, 10)
    return d & (d - 1) == 0


def _simple_end(onset: Fraction, raw_end: float, limit: Fraction | None,
                families: dict[int, str]) -> Fraction:
    """Nearest plausible written end when the note is followed by a rest."""
    best, best_cost = None, np.inf
    for d, pref in _SIMPLE:
        end = onset + d
        if limit is not None and end > limit:
            continue
        # The end must sit on the grid of the beat it falls in (thirds in a
        # triplet beat, fifths in a quintuplet beat, halves/quarters
        # otherwise); mixing grids inside one beat produces tuplets that
        # can't be notated.
        if not on_beat_grid(end, families):
            continue
        cost = abs(raw_end - float(end)) / 0.25 + pref
        if limit is not None:
            # The rest that follows should be simple too: a 16th note then a
            # double-dotted rest is worse than an eighth then an eighth rest.
            rest = limit - end
            if rest and not any(rest == d for d, _ in _SIMPLE):
                cost += 1.0
        if cost < best_cost:
            best, best_cost = end, cost
    if best is None:  # not even a sixteenth fits before the next note
        return limit
    return best


def resolve_lengths(voice: Voice, families: dict[int, str], bar_lines: list[int],
                    decaying: bool = True) -> None:
    """Decide where each note in the voice stops (in place).

    A note isn't stretched across a bar line just to reach the next note when
    its sound had already died before the bar line: that would add a tie the
    composer almost certainly didn't write (e.g. waltz chords on beat 3).
    """
    legato_frac = 0.4 if decaying else 0.75
    bars = np.array(sorted(bar_lines))
    events = voice.events
    for i, e in enumerate(events):
        nxt = events[i + 1].offset if i + 1 < len(events) else None
        raw = max(e.raw_end, float(e.offset))
        if nxt is not None:
            k = np.searchsorted(bars, float(e.offset), side="right")
            if k < len(bars) and e.offset < bars[k] < nxt and raw < bars[k] + 0.25:
                nxt = Fraction(int(bars[k]))
        if nxt is None:
            end = _simple_end(e.offset, raw, None, families)
        else:
            ioi = float(nxt - e.offset)
            sounded = raw - float(e.offset)
            if raw >= float(nxt) - 0.125 or sounded >= legato_frac * ioi:
                end = nxt  # held (or rung) until the next note of this line
            else:
                end = _simple_end(e.offset, raw, nxt, families)
        for q in e.notes:
            q.duration = end - e.offset


def role_names(n: int) -> list[str]:
    if n == 1:
        return ["Piano"]
    if n == 2:
        return ["Melody", "Bass"]
    middle = ["Accompaniment"] if n == 3 else [f"Voice {k}" for k in range(2, n)]
    return ["Melody", *middle, "Bass"]
