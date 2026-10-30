"""Turn quantized parts into a notated music21 score.

Everything upstream works in *beats* (fractions of the tracked pulse). Here
beats become quarter lengths (1 beat = a quarter, or a dotted quarter in
compound meters), bars become measures with time-signature changes, and
tempo events, chord symbols, grace notes and trills become notation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from music21 import (chord, clef, duration, expressions, harmony, instrument, key,
                     layout, metadata, meter, note, pitch, stream, tempo, tie)

from .beats import Bar, TempoEvent
from .harmony import Chord, Key
from .rhythm import QuantizedNote
from .safety import ensure_valid

# Notes at or above middle C go to the treble staff in a grand staff.
SPLIT_POINT = 60


@dataclass
class PartSpec:
    name: str
    instrument: str               # music21.instrument class name, e.g. "Piano"
    notes: list[QuantizedNote]
    grand_staff: bool | None = None  # None = decide from range


@dataclass
class ScoreLayout:
    bars: list[Bar]
    compound: bool = False        # beat = dotted quarter (6/8, 9/8, 12/8)
    tempo: list[TempoEvent] | None = None
    chords: list[tuple[int, Chord]] | None = None  # (beat, chord) at each change

    @property
    def beat_ql(self) -> Fraction:
        return Fraction(3, 2) if self.compound else Fraction(1)

    def time_signature(self, beats: int) -> str:
        return f"{3 * beats}/8" if self.compound else f"{beats}/4"


def to_music21_key(k: Key) -> key.Key:
    name = k.name(k.tonic)
    return key.Key(name[0] + name[1:].replace("b", "-"), "minor" if k.minor else "major")


def spell(midi: int, k: key.Key) -> pitch.Pitch:
    """Choose the enharmonic spelling that fits the key (Bb in F major, not A#)."""
    p = pitch.Pitch(midi=midi)
    for sp in k.getPitches():
        if sp.pitchClass == p.pitchClass:
            p.name = sp.name
            return p
    if k.sharps < 0 and p.accidental is not None and p.accidental.name == "sharp":
        p = p.getEnharmonic()
    return p


def _dyadic(x: Fraction) -> bool:
    d = Fraction(x).denominator
    return d & (d - 1) == 0


def _expressible(ql: Fraction) -> bool:
    """Can this length be written as one note (possibly dotted / in one tuplet)?"""
    d = duration.Duration(ql)
    return d.type not in ("complex", "inexpressible") and len(d.tuplets) <= 1


def _split_points(a: Fraction, b: Fraction, beat_ql: Fraction) -> list[Fraction]:
    """Interior points that cut [a, b) (in beats) into notatable pieces."""
    if _expressible((b - a) * beat_ql):
        return []
    unit = Fraction(1, max(a.denominator, b.denominator, 1))
    pts, cur = [], a
    while cur < b:
        nxt = b
        while nxt > cur + unit and not _expressible((nxt - cur) * beat_ql):
            nxt -= unit
        if nxt < b:
            pts.append(nxt)
        cur = nxt
    return pts


def _ceil(x: Fraction) -> int:
    return -(-x.numerator // x.denominator)


def _segments(qnotes: list[QuantizedNote], start: Fraction, end: Fraction,
              bar_lines: set[Fraction], beat_ql: Fraction, k: key.Key):
    """Split overlapping notes into one voice of chords/rests joined by ties.

    Every onset, release and bar line is a boundary. Spans involving tuplet
    positions are also cut at beat lines so a tuplet never straddles a beat,
    and any length that can't be one written note is split into tied pieces.
    """
    notes = [q for q in qnotes if q.offset < end and q.offset + q.duration > start]
    points = {start, end} | {b for b in bar_lines if start < b < end}
    for q in notes:
        points.add(max(q.offset, start))
        points.add(min(q.offset + q.duration, end))
    raw = sorted(points)
    for a, b in zip(raw, raw[1:]):
        if not (_dyadic(a) and _dyadic(b)):
            points |= {Fraction(x) for x in range(int(a) + 1, _ceil(b))}
    raw = sorted(points)
    for a, b in zip(raw, raw[1:]):
        points |= set(_split_points(a, b, beat_ql))
    bounds = sorted(points)

    for a, b in zip(bounds, bounds[1:]):
        ql = (b - a) * beat_ql
        active = [q for q in notes if q.offset <= a < q.offset + q.duration]
        if not active:
            yield a, note.Rest(quarterLength=ql), []
            continue
        pieces, graces, trill = [], [], False
        for q in sorted(active, key=lambda q: q.pitch):
            n = note.Note(spell(q.pitch, k), quarterLength=ql)
            starts_here = q.offset == a
            ends_here = q.offset + q.duration == b
            if not (starts_here and ends_here):
                n.tie = tie.Tie("start" if starts_here else
                                "stop" if ends_here else "continue")
            if starts_here:
                graces.extend(q.graces)
                trill = trill or q.trill
            pieces.append(n)
        el = pieces[0] if len(pieces) == 1 else chord.Chord(pieces, quarterLength=ql)
        if trill:
            el.expressions.append(expressions.Trill())
        yield a, el, graces


def _grace(p: int, k: key.Key) -> note.Note:
    g = note.Note(spell(p, k), type="eighth").getGrace()
    g.duration.slash = True  # acciaccatura
    return g


def _instrument(name: str) -> instrument.Instrument:
    inst = getattr(instrument, name, instrument.Piano)()
    # Keep everything at concert pitch; avoids octave-transposition surprises
    # for guitar and bass in MusicXML import.
    inst.transposition = None
    return inst


def _staff(qnotes, layout_: ScoreLayout, window: list[Bar], k: key.Key,
           staff_clef: clef.Clef, cls=stream.Part, pickup_full: int | None = None) -> stream.Part:
    """One staff of measures. With `pickup_full`, window[0] is a shortened
    pickup bar of a `pickup_full`-beat meter (measure 0, no leading rests)."""
    p = cls()
    beat_ql = layout_.beat_ql
    start = Fraction(window[0].start)
    end = Fraction(window[-1].start + window[-1].beats)
    bar_lines = {Fraction(b.start) for b in window}
    elements = list(_segments(qnotes, start, end, bar_lines, beat_ql, k))
    i = 0
    def meter_beats(bi: int) -> int:
        return pickup_full if (bi == 0 and pickup_full) else window[bi].beats

    for bi, bar in enumerate(window):
        m = stream.Measure(number=bi if pickup_full else bi + 1)
        if bi == 0:
            m.clef = staff_clef
            m.keySignature = key.KeySignature(k.sharps)
            if pickup_full:
                m.paddingLeft = float((pickup_full - bar.beats) * beat_ql)
        if bi == 0 or meter_beats(bi) != meter_beats(bi - 1):
            m.timeSignature = meter.TimeSignature(layout_.time_signature(meter_beats(bi)))
        bar_end = bar.start + bar.beats
        while i < len(elements) and elements[i][0] < bar_end:
            off, el, graces = elements[i]
            pos = float((off - bar.start) * beat_ql)
            for gp in graces:
                g = _grace(gp, k)
                g.priority = -1
                m.insert(pos, g)
            m.insert(pos, el)
            i += 1
        p.append(m)
    for m in p.getElementsByClass(stream.Measure):
        try:
            m.makeBeams(inPlace=True)
        except Exception:
            pass  # beaming is cosmetic; MuseScore re-beams on import anyway
    return p


def _abbreviation(name: str) -> str:
    if name.startswith("Voice "):
        return "V. " + name.split()[1]
    return {"Melody": "Mel.", "Accompaniment": "Acc.", "Bass": "Bass"}.get(name, name[:4] + ".")


def _clef_for(qnotes: list[QuantizedNote]) -> clef.Clef:
    s = stream.Stream()
    for q in qnotes:
        s.append(note.Note(q.pitch))
    return clef.bestClef(s, recurse=True) if len(s) else clef.TrebleClef()


def _bar_window(parts: list[PartSpec], bars: list[Bar]) -> list[Bar]:
    """The bars from the one holding the first note to the one holding the last."""
    notes = [q for p in parts for q in p.notes]
    if not notes:
        return bars[:1]
    first = min(q.offset for q in notes)
    last = max(q.offset + q.duration for q in notes)
    i0 = max((i for i, b in enumerate(bars) if b.start <= first), default=0)
    i1 = next((i for i, b in enumerate(bars) if b.start + b.beats >= last), len(bars) - 1)
    return bars[i0: i1 + 1]


def _add_marks(part: stream.Part, layout_: ScoreLayout, window: list[Bar], k: Key) -> None:
    """Tempo marks and chord symbols go on the top staff."""
    measures = list(part.getElementsByClass(stream.Measure))
    beat_ql = layout_.beat_ql
    referent = duration.Duration(1.5 if layout_.compound else 1.0)

    def place(beat: int):
        for m, bar in zip(measures, window):
            if bar.start <= beat < bar.start + bar.beats:
                return m, float((beat - bar.start) * beat_ql)
        return None, 0.0

    for ev in layout_.tempo or []:
        m, pos = place(max(ev.beat, window[0].start))
        if m is None:
            continue
        bpm = round(ev.bpm)  # beats per minute; the referent is the beat unit
        if ev.kind in ("tempo", "a tempo"):
            m.insert(pos, tempo.MetronomeMark(
                number=bpm, referent=referent,
                text="a tempo" if ev.kind == "a tempo" else None))
        else:
            m.insert(pos, expressions.TextExpression(ev.kind))

    for beat, ch in layout_.chords or []:
        m, pos = place(beat)
        if m is None or ch is None:
            continue
        root = k.name(ch.root)
        try:
            m.insert(pos, harmony.ChordSymbol(root[0] + root[1:].replace("b", "-")
                                              + ch.quality))
        except Exception:
            pass


def build_score(parts: list[PartSpec], layout_: ScoreLayout, k: Key,
                title: str = "Transcription") -> stream.Score:
    score = stream.Score()
    score.insert(0, metadata.Metadata(title=title, composer="Transcribed"))
    window = _bar_window(parts, layout_.bars)
    # Music that starts after the first bar line gets a pickup measure
    # (anacrusis) instead of a bar padded with rests.
    pickup_full = None
    notes = [q for spec in parts for q in spec.notes]
    if notes:
        first = int(min(q.offset for q in notes))  # beat holding the first note
        if first > window[0].start:
            full = window[0]
            window = [Bar(first, full.start + full.beats - first)] + window[1:]
            pickup_full = full.beats
    mk = to_music21_key(k)
    top: stream.Part | None = None

    for spec in parts:
        pitches = [q.pitch for q in spec.notes] or [SPLIT_POINT]
        grand = spec.grand_staff
        if grand is None:
            grand = (spec.instrument == "Piano"
                     and min(pitches) < 55 and max(pitches) > 65)
        if grand:
            upper = [q for q in spec.notes if q.pitch >= SPLIT_POINT]
            lower = [q for q in spec.notes if q.pitch < SPLIT_POINT]
            rh = _staff(upper, layout_, window, mk, clef.TrebleClef(), stream.PartStaff, pickup_full)
            lh = _staff(lower, layout_, window, mk, clef.BassClef(), stream.PartStaff, pickup_full)
            rh.insert(0, _instrument(spec.instrument))
            rh.partName = spec.name
            score.insert(0, rh)
            score.insert(0, lh)
            score.insert(0, layout.StaffGroup([rh, lh], name=spec.name,
                                              symbol="brace", barTogether=True))
            top = top or rh
        else:
            p = _staff(spec.notes, layout_, window, mk, _clef_for(spec.notes),
                       pickup_full=pickup_full)
            inst = _instrument(spec.instrument)
            # Name the staff by its role on every system ("Mel.", "V. 2"),
            # not by the instrument, since every part may be a piano.
            inst.partName = spec.name
            inst.partAbbreviation = _abbreviation(spec.name)
            p.insert(0, inst)
            p.partName = spec.name
            p.partAbbreviation = inst.partAbbreviation
            score.insert(0, p)
            top = top or p
    if top is not None:
        _add_marks(top, layout_, window, k)
    # Last line of defence: any measure that isn't cleanly notatable is
    # rebuilt on a plain grid, so the file always opens in MuseScore.
    score.beat_ql = layout_.beat_ql
    score.repaired_measures = ensure_valid(score, layout_.beat_ql)
    return score
