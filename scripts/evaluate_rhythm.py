"""Benchmark the *notation* side: rhythm, meter, tempo changes, ornaments.

Each test piece is written as notation (beats, durations, pitches, meter),
rendered through a tempo curve (steady, ritardando, accelerando) with a
harmonic synth, then transcribed. We report:

  rhythm  - share of notes whose written onset (in beats, relative to the
            first note) is exactly right
  durs    - share with exactly the right written duration
  meter   - detected beats per bar / compound flag vs truth
  downbeat- whether bar lines fall in the right place
  tempo   - tempo marks found (rit./accel./a tempo)
  ornaments - grace notes / trills recognised

    python scripts/evaluate_rhythm.py [--out-dir eval_out/rhythm]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcriber.export import export  # noqa: E402
from transcriber.pipeline import TranscribeOptions, transcribe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SR, synth_note  # noqa: E402


@dataclass
class Piece:
    name: str
    beats_per_bar: int
    notes: list[tuple[F, F, int]]          # (onset beat, duration beats, pitch)
    bpm: float = 100.0
    compound: bool = False
    tempo_curve: callable = None           # beat -> bpm multiplier
    graces: list[tuple[F, int]] = field(default_factory=list)  # (beat, pitch) before a note
    trills: list[tuple[F, F, int]] = field(default_factory=list)  # (beat, dur, lower pitch)
    expect_tempo: tuple[str, ...] = ()
    voices: list[list[tuple[F, F, int]]] = field(default_factory=list)  # truth, top first
    pickup: F = F(0)                       # beats before the first downbeat


def beat_to_time(piece: Piece, beats: np.ndarray) -> np.ndarray:
    """Integrate the tempo curve: seconds at each beat position."""
    grid = np.linspace(0, float(max(beats.max(), 1)) + 1, 4000)
    mult = np.array([piece.tempo_curve(b) if piece.tempo_curve else 1.0 for b in grid])
    sec_per_beat = 60.0 / (piece.bpm * mult)
    t = np.concatenate([[0], np.cumsum(sec_per_beat[:-1] * np.diff(grid))])
    return np.interp(beats, grid, t) + 1.0  # 1 s lead-in


def render(piece: Piece) -> np.ndarray:
    events = []
    for on, dur, p in piece.notes:
        t0, t1 = beat_to_time(piece, np.array([float(on), float(on + dur)]))
        events.append((t0, (t1 - t0) * 0.92, p, 0.8))
    for on, p in piece.graces:
        t0 = beat_to_time(piece, np.array([float(on)]))[0]
        events.append((t0 - 0.07, 0.055, p, 0.6))
    for on, dur, p in piece.trills:
        t0, t1 = beat_to_time(piece, np.array([float(on), float(on + dur)]))
        k, t = 0, t0
        while t < t1 - 0.05:
            events.append((t, 0.065, p + (2 if k % 2 else 0), 0.6))
            t += 0.07
            k += 1
    end = max(t + d for t, d, *_ in events) + 1.5
    out = np.zeros(int(end * SR))
    for t, d, p, v in events:
        s = synth_note(p, d, v)
        i = int(t * SR)
        out[i:i + len(s)] += s[: len(out) - i]
    out /= np.abs(out).max()
    rng = np.random.default_rng(1)
    return (0.8 * out + 0.003 * rng.normal(size=len(out))).astype(np.float32)


def _seq(start: F, items, pitches) -> list[tuple[F, F, int]]:
    out, t = [], start
    for d, p in zip(items, pitches):
        if p is not None:
            out.append((t, F(d), p))
        t += F(d)
    return out


def pieces() -> list[Piece]:
    t3 = F(1, 3)
    # 1) 4/4 melody mixing eighths and triplets over a bass on each downbeat.
    mel = _seq(F(0), [1, F(1, 2), F(1, 2), t3, t3, t3, 1,
                      F(1, 2), F(1, 2), 1, t3, t3, t3, 1,
                      F(3, 4), F(1, 4), F(1, 2), F(1, 2), t3, t3, t3, 1,
                      2, 2],
               [72, 74, 76, 77, 76, 74, 72, 71, 72, 74, 76, 77, 79, 77,
                76, 74, 72, 71, 72, 74, 76, 72, 67, 72])
    bass = [(F(4 * i), F(4), p) for i, p in enumerate([48, 53, 55, 48, 43, 48])]
    p1 = Piece("triplets_4-4", 4, sorted(mel + bass), bpm=96, voices=[mel, bass])

    # 2) Waltz in 3/4: bass on 1, chords on 2 and 3, chord changes every bar.
    prog = [(48, (55, 64)), (53, (57, 65)), (55, (59, 65)), (48, (55, 64))] * 2
    w_bass, w_chords, w_mel = [], [], []
    for bar, (b, ch) in enumerate(prog):
        s = F(3 * bar)
        w_bass.append((s, F(1), b))
        for beat in (1, 2):
            for p in ch:
                w_chords.append((s + beat, F(1), p))
        w_mel.append((s, F(2), [72, 77, 74, 72, 76, 77, 79, 72][bar]))
        w_mel.append((s + 2, F(1), [74, 76, 72, 71, 74, 76, 74, 72][bar]))
    p2 = Piece("waltz_3-4", 3, sorted(w_bass + w_chords + w_mel), bpm=120,
               voices=[w_mel, w_chords, w_bass])

    # 3) Steady eighths then a ritardando over the last two bars.
    line = _seq(F(0), [F(1, 2)] * 40, [60, 64, 67, 72, 67, 64] * 6 + [60, 62, 64, 60])
    bass3 = [(F(4 * i), F(4), 48 if i % 2 == 0 else 43) for i in range(5)]
    p3 = Piece("ritardando", 4, sorted(line + bass3), bpm=110,
               tempo_curve=lambda b: 1.0 if b < 12 else max(0.6, 1 - 0.05 * (b - 12)),
               expect_tempo=("rit.",), voices=[line, bass3])

    # 4) 6/8 (compound): eighths in groups of three.
    six = _seq(F(0), [t3] * 36 + [F(1)] * 2,
               [67, 71, 74, 72, 71, 69, 67, 69, 71, 72, 74, 76] * 3 + [79, 67])
    bass6 = [(F(2 * i), F(2), [43, 50, 48, 50, 43, 50, 43][i]) for i in range(7)]
    p4 = Piece("compound_6-8", 2, sorted(six + bass6), bpm=66, compound=True,
               voices=[six, bass6])

    # 5) Grace notes and a trill in a simple 4/4 tune.
    tune = _seq(F(0), [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 4],
                [67, 72, 71, 72, 74, 72, 76, 74, 72, 71, 72])
    bass5 = [(F(4 * i), F(4), p) for i, p in enumerate([48, 55, 48, 43, 48])]
    p5 = Piece("ornaments", 4, sorted(tune + bass5), bpm=84,
               graces=[(F(1), 71), (F(8), 74)],
               trills=[])
    # Replace the long note in bar 4 with a trill on D (D-E).
    p5.notes = [n for n in p5.notes if not (n[0] == F(10) and n[2] == 72)]
    p5.trills = [(F(10), F(2), 74)]
    p5.notes = [n for n in p5.notes if not (n[0] == F(11))]
    p5.voices = [[n for n in p5.notes if n[2] >= 60], bass5]
    # 6) Meter change: four bars of 4/4, then four bars of 3/4.
    mc, t = [], F(0)
    for bar in range(8):
        n = 4 if bar < 4 else 3
        root = [48, 53, 55, 48, 45, 50, 43, 48][bar]
        mc.append((t, F(n), root))
        mc += [(t + i, F(1), 72 + [0, 2, 4, 5, 7][(bar + i) % 5]) for i in range(n)]
        t += n
    p6 = Piece("meter_change", 4, sorted(mc), bpm=100,
               voices=[[n for n in mc if n[2] >= 60], [n for n in mc if n[2] < 60]])
    p6.bar_lengths = [4] * 4 + [3] * 4

    # 7) Sudden tempo change (90 -> 130 BPM at bar 5), then an accelerando.
    st = _seq(F(0), [F(1, 2)] * 64, ([60, 64, 67, 64] * 16))
    # Bass on every beat, as in most real music, so the new pulse is audible.
    bass7 = [(F(i), F(1), 48 if (i // 4) % 2 == 0 else 43) for i in range(32)]
    p7 = Piece("tempo_change", 4, sorted(st + bass7), bpm=90,
               tempo_curve=lambda b: 1.0 if b < 16 else (1.45 if b < 24 else 1.45 + 0.06 * (b - 24)),
               expect_tempo=("tempo", "accel."), voices=[st, bass7])
    # 8) Fast runs: 16th and 32nd scales over held chords and a bass line.
    scale = [72, 74, 76, 77, 79, 81, 83, 84, 86, 84, 83, 81, 79, 77, 76, 74]
    fast = _seq(F(0), [1, 1, 1, 1] + [F(1, 4)] * 16 + [F(1, 8)] * 16 + [2] + [4],
                [72, 76, 79, 76] + scale + scale + [72] + [72])
    chords8 = [(F(2 * i), F(2), p) for i in range(8)
               for p in [(55, 64), (57, 65), (55, 62), (55, 64)][i // 2]]
    bass8 = [(F(4 * i), F(4), p) for i, p in enumerate([48, 53, 43, 48])]
    p8 = Piece("fast_runs", 4, sorted(fast + chords8 + bass8), bpm=92,
               voices=[fast, chords8, bass8])

    # 9) Syncopation: eighth-quarter-eighth figures and a tie across the bar.
    syn = _seq(F(0), [F(1, 2), 1, F(1, 2), F(1, 2), F(1, 2), 1,
                      F(1, 2), 1, F(1, 2), 1, F(3, 2) + 1,  # last one tied over the bar
                      F(1, 2), F(1, 2), F(1, 2), 1, F(1, 2), 2, 2],
               [72, 74, 76, 74, 72, 71, 72, 74, 76, 77, 79,
                77, 76, 74, 72, 74, 72, 67])
    bass9 = [(F(2 * i), F(2), [48, 55, 53, 55, 48, 55, 43, 48][i]) for i in range(8)]
    p9 = Piece("syncopation", 4, sorted(syn + bass9), bpm=100, voices=[syn, bass9])

    # 10) Pickup: a 3/4 tune starting on beat 3 (one-beat anacrusis).
    tune10 = _seq(F(0), [1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3],
                  [67, 72, 74, 72, 77, 76, 72, 67, 74, 72, 72])
    bass10 = [(F(1 + 3 * i), F(3), p) for i, p in enumerate([48, 48, 53, 55, 48])]
    p10 = Piece("pickup_3-4", 3, sorted(tune10 + bass10), bpm=110, pickup=F(1),
                voices=[tune10, bass10])
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]


def score_piece(piece: Piece, result) -> dict:
    est = sorted((q for p in result.parts for q in p.quantized),
                 key=lambda q: (q.offset, q.pitch))
    ref = sorted(piece.notes, key=lambda n: (n[0], n[2]))
    if not est:
        return {"rhythm": 0.0, "durs": 0.0}
    # Align by the first note (both sides measure in beats).
    shift = est[0].offset - ref[0][0]
    est_set = {(q.offset - shift, q.pitch): q.duration for q in est}
    hits = [(on, p) in est_set for on, _, p in ref]
    dur_hits = [est_set.get((on, p)) == d for on, d, p in ref]
    # Downbeats: is the bar starting at the first note where truth says?
    bar_starts = {b.start for b in result.bars}
    last = max(on for on, _, _ in piece.notes)
    lengths = getattr(piece, "bar_lengths", None) or \
        [piece.beats_per_bar] * (int(last // piece.beats_per_bar) + 1)
    starts = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(int)
    truth_bars = {int(shift + piece.pickup) + int(s) for s in starts}
    downbeat_ok = len(truth_bars & bar_starts) / len(truth_bars)
    window = [b for b in result.bars if shift <= b.start <= shift + last]
    meters = [b.beats for b in window] or [0]
    seq = [m for i, m in enumerate(meters) if i == 0 or m != meters[i - 1]]
    meter = "->".join(map(str, seq))
    marks = [e.kind for e in result.tempo]
    # Voices: label every estimated part by its majority true voice, then
    # count notes that sit in a part labelled with their own true voice.
    truth = {(on, p): vi for vi, v in enumerate(piece.voices) for on, _, p in v}
    hit = total = 0
    for part in result.parts:
        labels = [truth.get((q.offset - shift, q.pitch)) for q in part.quantized]
        labels = [x for x in labels if x is not None]
        if labels:
            major = max(set(labels), key=labels.count)
            hit += sum(x == major for x in labels)
            total += len(labels)
    voice_acc = hit / total if total else 0.0
    # Ties per written note: ideally only notes crossing a bar line are tied.
    starts = ties = 0
    for part in result.score.parts:
        for n in part.recurse().notes:
            if n.tie is None or n.tie.type == "start":
                starts += 1
            if n.tie is not None and n.tie.type == "start":
                ties += 1
    # Tuplet positions the true score doesn't have.
    ref_on = {on for on, _, _ in ref}
    fake_tuplets = len({q.offset - shift for q in est
                        if (q.offset - shift).denominator % 3 == 0
                        or (q.offset - shift).denominator % 5 == 0} - ref_on)
    # Pickup bar written as a real (shortened) first measure?
    first = result.score.parts[0].getElementsByClass("Measure")[0]
    pickup_ok = (piece.pickup == 0 and first.paddingLeft == 0) or \
        (piece.pickup > 0 and first.paddingLeft > 0)
    trills = sum(q.trill for q in est)
    graces = sum(len(q.graces) for q in est)
    return {
        "rhythm": float(np.mean(hits)), "durs": float(np.mean(dur_hits)),
        "meter": f"{meter}{'c' if result.compound else ''} vs "
                 f"{'->'.join(str(m) for i, m in enumerate(lengths) if i == 0 or m != lengths[i - 1])}"
                 f"{'c' if piece.compound else ''}",
        "downbeats": downbeat_ok,
        # Every mark after the opening tempo.
        "tempo": ",".join(f"{e.kind}@{round(e.bpm)}" if e.kind == "tempo" else e.kind
                          for e in result.tempo[1:]) or "-",
        "voices": f"{len(result.parts)}/{len(piece.voices)}",
        "voice_acc": voice_acc,
        "ties": ties / max(starts, 1),
        "fake_tuplets": fake_tuplets,
        "pickup": pickup_ok,
        "graces": f"{graces}/{len(piece.graces)}",
        "trills": f"{trills}/{len(piece.trills)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="eval_out/rhythm")
    ap.add_argument("--only")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for piece in pieces():
        if args.only and args.only != piece.name:
            continue
        wav = out / f"{piece.name}.wav"
        sf.write(wav, render(piece), SR)
        result = transcribe(wav, TranscribeOptions())
        export(result.score, out / f"{piece.name}.musicxml")
        try:
            export(result.score, out / f"{piece.name}.png")
        except RuntimeError as e:
            print("  render failed:", e)
        s = score_piece(piece, result)
        rows.append((piece.name, s))
        print(f"{piece.name:14s} rhythm={s['rhythm']:.2f} durs={s['durs']:.2f} "
              f"voices={s['voices']} voice_acc={s['voice_acc']:.2f} ties/note={s['ties']:.2f} "
              f"fake_tuplets={s['fake_tuplets']} pickup={'ok' if s['pickup'] else 'NO'} "
              f"meter={s['meter']} downbeats={s['downbeats']:.2f} "
              f"tempo={s['tempo']} graces={s['graces']} trills={s['trills']}", flush=True)
    if rows:
        for k in ("rhythm", "durs", "voice_acc", "ties"):
            print(f"mean {k}: {np.mean([s[k] for _, s in rows]):.2f}")


if __name__ == "__main__":
    main()
