"""Recognise trills and grace notes in detected notes before quantization.

Both are very fast notes. Left in place, a quantizer has to explain them with
32nd/64th-note rhythms that no human would write; folding them into their
main note (as a trill mark or a grace note) keeps the rhythm clean.
"""

from __future__ import annotations

from .notes import Note

TRILL_MAX_NOTE_S = 0.16   # each trill note is at most this long
TRILL_MAX_GAP_S = 0.06
TRILL_MIN_NOTES = 4
GRACE_MAX_S = 0.16        # detected length (the model stretches very short notes)
GRACE_MAX_LEAD_S = 0.12   # grace onset to main-note onset...
GRACE_MAX_LEAD_BEATS = 0.2  # ...and at most this much of a beat, so 16ths stay notes
GRACE_MAX_INTERVAL = 5    # semitones; graces are almost always close neighbours


def find_trills(notes: list[Note]) -> list[Note]:
    """Collapse fast alternations between two neighbouring pitches into one trilled note."""
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    used = [False] * len(notes)
    out: list[Note] = []
    for i, n in enumerate(notes):
        if used[i]:
            continue
        if n.duration > TRILL_MAX_NOTE_S:
            continue
        # Follow a chain of short notes alternating between two pitches.
        chain = [i]
        pitches = {n.pitch}
        last = n
        while True:
            nxt = None
            for j in range(chain[-1] + 1, len(notes)):
                m = notes[j]
                if m.start > last.end + TRILL_MAX_GAP_S:
                    break
                if (not used[j] and m.start > last.start + 0.02
                        and m.pitch != last.pitch and abs(m.pitch - last.pitch) <= 2
                        and m.duration <= TRILL_MAX_NOTE_S * 1.5):
                    nxt = j
                    break
            if nxt is None or len(pitches | {notes[nxt].pitch}) > 2:
                break
            chain.append(nxt)
            pitches.add(notes[nxt].pitch)
            last = notes[nxt]
        if len(chain) >= TRILL_MIN_NOTES and len(pitches) == 2:
            for j in chain:
                used[j] = True
            members = [notes[j] for j in chain]
            # The principal note is the lower one; the trill adds the upper neighbour.
            out.append(Note(members[0].start, members[-1].end, min(pitches),
                            max(m.amplitude for m in members), trill=True))
    out.extend(n for i, n in enumerate(notes) if not used[i])
    return sorted(out, key=lambda n: (n.start, n.pitch))


def find_grace_notes(notes: list[Note], beat_seconds=None) -> list[Note]:
    """Attach very short notes that lead straight into a longer neighbour as grace notes.

    `beat_seconds(t)` gives the local beat length, so that at fast tempi a
    genuine sixteenth before a long note isn't mistaken for an ornament.
    """
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    graced: set[int] = set()
    for i, g in enumerate(notes):
        if g.duration > GRACE_MAX_S or g.trill:
            continue
        # Part of a chord or a fast run? Then it's a real note, not an ornament.
        if any(j != i and abs(o.start - g.start) < 0.03 for j, o in enumerate(notes)):
            continue
        # A short note right before it means we're inside a run (judged by
        # onsets, since run notes often ring into each other).
        if any(0.02 < g.start - o.start < 0.2 and o.duration < 0.25
               for o in notes if o is not g):
            continue
        lead = GRACE_MAX_LEAD_S
        if beat_seconds is not None:
            lead = min(lead, GRACE_MAX_LEAD_BEATS * beat_seconds(g.start))
        targets = [m for m in notes
                   if 0.02 < m.start - g.start <= lead
                   and abs(m.pitch - g.pitch) <= GRACE_MAX_INTERVAL
                   and m.pitch != g.pitch
                   and m.duration >= 3 * g.duration]
        if targets:
            main = min(targets, key=lambda m: m.start)
            main.graces.append(g.pitch)
            graced.add(i)
    return [n for i, n in enumerate(notes) if i not in graced]


def detect_ornaments(notes: list[Note], beat_seconds=None) -> list[Note]:
    return find_grace_notes(find_trills(notes), beat_seconds)
