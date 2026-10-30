"""Measure note accuracy on synthetic recordings with known ground truth.

Renders test pieces with a harmonic, decaying piano-like synth, adds noise,
runs the full pipeline, and reports mir_eval note precision/recall/F1 (a note
counts as correct if its pitch matches and its onset is within 50 ms).

    python scripts/evaluate.py [--snr 20] [--out-dir eval_out]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mir_eval
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcriber.audio import SAMPLE_RATE, preprocess  # noqa: E402
from transcriber.notes import detect_notes  # noqa: E402
from transcriber.pipeline import transcribe  # noqa: E402
from transcriber.export import export  # noqa: E402

SR = SAMPLE_RATE


def synth_note(midi: int, dur: float, vel: float = 0.8) -> np.ndarray:
    """Additive synthesis: 8 partials with 1/k rolloff and slight inharmonicity."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    tail = 0.3
    t = np.arange(int((dur + tail) * SR)) / SR
    sig = np.zeros_like(t)
    for k in range(1, 9):
        fk = f0 * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > SR / 2:
            break
        sig += np.sin(2 * np.pi * fk * t) / k * np.exp(-t * (1.5 + 0.6 * k))
    attack = np.minimum(1, t / 0.005)
    release = np.where(t > dur, np.exp(-(t - dur) * 25), 1.0)
    return vel * sig * attack * release


def render(notes, total: float) -> np.ndarray:
    out = np.zeros(int((total + 1) * SR))
    for start, end, pitch in notes:
        s = synth_note(pitch, end - start)
        i = int(start * SR)
        out[i:i + len(s)] += s[: len(out) - i]
    return out / np.max(np.abs(out)) * 0.8


def piece_melody(bpm=100):
    """C major melody with eighths and quarters (monophonic)."""
    b = 60 / bpm
    seq = [(60, 1), (62, 1), (64, 1), (65, 1), (67, 2), (67, 2),
           (69, .5), (69, .5), (69, .5), (69, .5), (67, 2),
           (65, 1), (65, 1), (64, 1), (64, 1), (62, 1), (62, 1), (60, 2)]
    t, notes = 0.0, []
    for p, beats in seq:
        notes.append((t, t + beats * b * 0.95, p))
        t += beats * b
    return notes, t


def piece_piano(bpm=90):
    """Melody over block chords and a bass line (polyphonic, grand staff)."""
    b = 60 / bpm
    chords = [[48, 55, 64], [53, 57, 65], [55, 59, 62], [48, 55, 64]]
    mel = [72, 74, 76, 77, 77, 76, 74, 72, 71, 72, 74, 79, 76, 74, 72, 72]
    notes = []
    for bar, ch in enumerate(chords):
        for p in ch:
            notes.append((bar * 4 * b, (bar * 4 + 4) * b * 0.98, p))
        for i in range(4):
            on = (bar * 4 + i) * b
            notes.append((on, on + b * 0.9, mel[bar * 4 + i]))
    return notes, 16 * b


def evaluate(name, notes, total, snr_db, out_dir: Path):
    clean = render(notes, total)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(clean))
    # Mix in some 60 Hz hum as well as broadband hiss.
    noise += 3 * np.sin(2 * np.pi * 60 * np.arange(len(clean)) / SR)
    noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2)) / 10 ** (snr_db / 20)
    # Two seconds of noise-only lead-in, like a real recording.
    lead = int(2 * SR)
    audio = np.concatenate([noise[:lead], clean + noise])
    wav = out_dir / f"{name}.wav"
    sf.write(wav, audio.astype(np.float32), SR)

    ref_iv = np.array([[s + 2, e + 2] for s, e, _ in notes])
    ref_hz = np.array([440 * 2 ** ((p - 69) / 12) for *_, p in notes])

    for denoise in (False, True):
        # Score raw detections against ground truth (before quantization).
        cleaned = preprocess(str(wav), use_denoise=denoise)
        det = detect_notes(cleaned, SR)
        # preprocess trims leading silence, so the time origin moves; try every
        # ref/est onset pairing as the alignment and keep the best one.
        est_iv0 = np.array([[n.start, n.end] for n in det]).reshape(-1, 2)
        est_hz = np.array([440 * 2 ** ((n.pitch - 69) / 12) for n in det])
        p = r = f = 0.0
        shifts = {round(a - b, 3) for a in ref_iv[:, 0] for b in est_iv0[:, 0]}
        for shift in (s for s in shifts if est_iv0.min() + s >= 0):
            scores = mir_eval.transcription.precision_recall_f1_overlap(
                ref_iv, ref_hz, est_iv0 + shift, est_hz, offset_ratio=None)
            if scores[2] > f:
                p, r, f, _ = scores
        print(f"{name:8s} snr={snr_db:>3}dB denoise={denoise!s:5}  "
              f"P={p:.2f} R={r:.2f} F1={f:.2f}  ({len(det)} est / {len(notes)} ref)")

    result = transcribe(wav)
    for ext in (".musicxml", ".mscz", ".pdf"):
        try:
            export(result.score, out_dir / f"{name}{ext}", result.notes)
        except RuntimeError as e:
            print("  export skipped:", e)
    print(f"  -> {out_dir / name}.* (tempo est. {result.bpm:.0f} BPM)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[30, 10])
    ap.add_argument("--out-dir", default="eval_out")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    for snr in args.snr:
        evaluate("melody", *piece_melody(), snr, out)
        evaluate("piano", *piece_piano(), snr, out)


if __name__ == "__main__":
    main()
