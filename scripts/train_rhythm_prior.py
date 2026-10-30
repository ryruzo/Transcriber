"""Learn a rhythm prior from PDMX (public-domain MuseScore scores, as MIDI).

For every part of every score we cut the music into metric beats (a quarter
in 4/4, a dotted quarter in 6/8, ...) and record the set of note-onset
positions inside each beat, e.g. (0,) for a quarter, (0, 1/2) for two
eighths, (0, 1/3, 2/3) for a triplet. Counting these over tens of thousands of
scores gives P(pattern), plus how beat "families" (duple / triplet /
quintuplet) follow each other. The quantizer uses this as its prior, so it
prefers rhythms that composers actually write.

    python scripts/train_rhythm_prior.py data/pdmx/mid --subset data/pdmx/subset_paths/deduplicated.txt \
        --max-files 40000 --out transcriber/models/rhythm_prior.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcriber.rhythm_prior import ALLOWED_DENOMINATORS, family, pattern_key  # noqa: E402


def _time_signatures(mid: mido.MidiFile) -> list[tuple[int, int, int]]:
    """(tick, numerator, denominator) for every time-signature event."""
    out = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "time_signature":
                out.append((tick, msg.numerator, msg.denominator))
    out.sort()
    return out or [(0, 4, 4)]


def _beats(mid: mido.MidiFile, end_tick: int) -> list[int]:
    """Start tick of every metric beat, honouring time-signature changes."""
    tpq = mid.ticks_per_beat
    sigs = _time_signatures(mid)
    beats = []
    for i, (start, num, den) in enumerate(sigs):
        stop = sigs[i + 1][0] if i + 1 < len(sigs) else end_tick
        compound = den == 8 and num in (6, 9, 12)
        beat_len = tpq * 3 // 2 if compound else tpq * 4 // den
        if beat_len <= 0:
            continue
        beats.extend(range(start, stop, beat_len))
    return beats


def process(path: str) -> tuple[Counter, Counter, int] | None:
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    patterns: Counter = Counter()
    transitions: Counter = Counter()
    skipped = 0
    tracks = []
    end_tick = 0
    for track in mid.tracks:
        tick, onsets = 0, []
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9:
                onsets.append(tick)
        end_tick = max(end_tick, tick)
        if onsets:
            tracks.append(sorted(set(onsets)))
    if not tracks:
        return None
    beats = _beats(mid, end_tick + 1)
    if len(beats) < 4:
        return None
    for onsets in tracks:
        prev_family = None
        j = 0
        for b, start in enumerate(beats):
            stop = beats[b + 1] if b + 1 < len(beats) else end_tick + 1
            while j < len(onsets) and onsets[j] < start:
                j += 1
            k = j
            pos = []
            while k < len(onsets) and onsets[k] < stop:
                pos.append(Fraction(onsets[k] - start, stop - start))
                k += 1
            if not pos:
                prev_family = None  # held note or rest: no rhythmic decision
                continue
            # MuseScore's swing playback (default 60%) exports swung eighths
            # at 3/5 of the beat; on the page they are plain eighths.
            if Fraction(3, 5) in pos and all(p in (0, Fraction(3, 5)) for p in pos):
                pos = [Fraction(1, 2) if p == Fraction(3, 5) else p for p in pos]
            if any(p.denominator not in ALLOWED_DENOMINATORS for p in pos):
                # Grace notes, swing playback or ornaments in the MIDI export.
                skipped += 1
                prev_family = None
                continue
            key = pattern_key(pos)
            patterns[key] += 1
            fam = family(pos)
            if prev_family is not None:
                transitions[f"{prev_family}>{fam}"] += 1
            prev_family = fam
    return patterns, transitions, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mid_dir")
    ap.add_argument("--subset", help="PDMX subset list (paths are matched by file stem)")
    ap.add_argument("--max-files", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="transcriber/models/rhythm_prior.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.mid_dir).rglob("*.mid"))
    if args.subset:
        keep = {Path(line.strip()).stem for line in open(args.subset, encoding="utf-8")}
        files = [f for f in files if Path(f).stem in keep]
    random.Random(0).shuffle(files)
    files = files[: args.max_files]
    print(f"Training on {len(files)} files")

    patterns: Counter = Counter()
    transitions: Counter = Counter()
    skipped = used = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(process, files, chunksize=64)):
            if res is not None:
                p, t, s = res
                patterns.update(p)
                transitions.update(t)
                skipped += s
                used += 1
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    total = sum(patterns.values())
    # Keep patterns seen at least 20 times; the long tail is mostly noise.
    kept = {k: c for k, c in patterns.most_common() if c >= 20}
    out = {
        "source": "PDMX (public-domain MuseScore scores), MIDI exports",
        "files": used,
        "beats": total,
        "irregular_beats_skipped": skipped,
        "patterns": kept,
        "family_transitions": dict(transitions.most_common()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{used} files, {total} beats, {len(kept)} patterns -> {args.out}")
    for k, c in list(kept.items())[:25]:
        print(f"  {c / total:7.3%}  {k or '(none)'}")


if __name__ == "__main__":
    main()
