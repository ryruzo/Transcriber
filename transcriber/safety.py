"""Make sure every measure is something MuseScore can import.

A transcription can still contain a measure that's mathematically fine but
not notatable (a tuplet straddling a beat, an incomplete triplet, a note
on a 1/12-beat position next to plain sixteenths). MuseScore rejects or
flags such scores as corrupted. Each measure is checked against strict
rules and, if it fails, rebuilt on a plain duple grid, which is always valid.
"""

from __future__ import annotations

from fractions import Fraction

from music21 import chord, duration, harmony, note, stream, tie

F = Fraction


def _content(m: stream.Measure):
    """Notes, chords and rests (no grace notes, chord symbols)."""
    return sorted((e for e in m.notesAndRests
                   if not e.duration.isGrace and not isinstance(e, harmony.Harmony)),
                  key=lambda e: e.offset)


def _q(x) -> Fraction:
    return Fraction(x).limit_denominator(960)


def measure_ok(m: stream.Measure, bar_ql: Fraction, beat_ql: Fraction) -> bool:
    pad = _q(m.paddingLeft)
    pos = F(0)
    tuplet_beats: dict[int, list] = {}
    for e in _content(m):
        off, ql = _q(e.offset), _q(e.quarterLength)
        if off != pos or ql <= 0:
            return False  # gap or overlap: the measure wouldn't add up
        d = e.duration
        if d.type in ("complex", "inexpressible", "zero") or len(d.tuplets) > 1:
            return False
        start = off + pad  # position within the full bar, for beat lines
        beat = int(start // beat_ql)
        if d.tuplets:
            if start + ql > (beat + 1) * beat_ql:
                return False  # tuplet crosses a beat line
            tuplet_beats.setdefault(beat, []).append((d.tuplets[0], ql))
        else:
            rel = start - beat * beat_ql
            if rel.denominator & (rel.denominator - 1):
                return False  # plain note on a tuplet position
        pos = off + ql
    if pos != bar_ql - pad:
        return False
    for beat, items in tuplet_beats.items():
        ratios = {(t.numberNotesActual, t.numberNotesNormal) for t, _ in items}
        if len(ratios) > 1 or sum(ql for _, ql in items) != beat_ql:
            return False  # mixed or incomplete tuplet group
    return True


def _pieces(a: Fraction, b: Fraction, beat_ql: Fraction, grid: Fraction) -> list[Fraction]:
    """Cut [a, b) at beat lines into notatable pieces on `grid`."""
    points = {a, b}
    k = (a // beat_ql + 1) * beat_ql
    while k < b:
        points.add(k)
        k += beat_ql
    pts = sorted(points)
    out = [pts[0]]
    for x, y in zip(pts, pts[1:]):
        cur = x
        while cur < y:
            nxt = y
            while nxt - cur > grid and duration.Duration(nxt - cur).type in ("complex", "inexpressible"):
                nxt -= grid
            out.append(nxt)
            cur = nxt
    return out


def sanitize_measure(m: stream.Measure, bar_ql: Fraction, beat_ql: Fraction) -> None:
    """Rebuild the measure's notes on a plain grid (16ths, or 8ths in compound time)."""
    pad = _q(m.paddingLeft)
    grid = F(1, 2) if beat_ql == F(3, 2) else F(1, 4)
    length = bar_ql - pad
    events = []  # (start, end, pitches, tie_in, tie_out)
    for e in _content(m):
        if e.isRest:
            continue
        a = round(_q(e.offset) / grid) * grid
        b = round((_q(e.offset) + _q(e.quarterLength)) / grid) * grid
        b = min(b, length)
        if b <= a:
            b = min(a + grid, length)
            if b <= a:
                continue
        pitches = [p.pitch for p in e.notes] if isinstance(e, chord.Chord) else [e.pitch]
        t = e.tie.type if e.tie is not None else (
            e.notes[0].tie.type if isinstance(e, chord.Chord) and e.notes[0].tie else None)
        events.append([a, b, pitches, t in ("stop", "continue"), t in ("start", "continue")])
    # One line: each event ends where the next begins at the latest.
    events.sort(key=lambda x: x[0])
    merged = []
    for ev in events:
        if merged and ev[0] == merged[-1][0]:
            merged[-1][2] = list({p.midi: p for p in merged[-1][2] + ev[2]}.values())
            merged[-1][1] = max(merged[-1][1], ev[1])
            continue
        if merged and merged[-1][1] > ev[0]:
            merged[-1][1] = ev[0]
        merged.append(ev)
    for e in list(_content(m)):
        m.remove(e)
    pos = F(0)

    def put(a, b, make):
        cuts = _pieces(a + pad, b + pad, beat_ql, grid)
        for i, (x, y) in enumerate(zip(cuts, cuts[1:])):
            el = make(i, len(cuts) - 1)
            el.duration = duration.Duration(y - x)
            m.insert(float(x - pad), el)

    for a, b, pitches, tie_in, tie_out in merged:
        if a > pos:
            put(pos, a, lambda i, n: note.Rest())

        def make(i, n, pitches=pitches, tie_in=tie_in, tie_out=tie_out):
            el = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            into = tie_in if i == 0 else True
            out = tie_out if i == n - 1 else True
            if into or out:
                el.tie = tie.Tie("continue" if into and out else "stop" if into else "start")
            return el
        put(a, b, make)
        pos = b
    if pos < length:
        put(pos, length, lambda i, n: note.Rest())


def ensure_valid(score: stream.Score, beat_ql: Fraction, force: bool = False) -> int:
    """Check every measure; rebuild the bad ones. Returns how many were fixed.

    With `force`, every measure containing tuplets is rebuilt too (last-resort
    mode when MuseScore still refuses the file).
    """
    fixed = 0
    for part in score.parts:
        for m in part.getElementsByClass(stream.Measure):
            ts = m.getContextByClass("TimeSignature")
            if ts is None:
                continue
            bar_ql = _q(ts.barDuration.quarterLength)
            has_tuplets = any(e.duration.tuplets for e in _content(m))
            if (force and has_tuplets) or not measure_ok(m, bar_ql, beat_ql):
                sanitize_measure(m, bar_ql, beat_ql)
                fixed += 1
    return fixed
