"""Command line: python -m transcriber recording.wav -o score.pdf -o score.mscz"""

from __future__ import annotations

import argparse
from pathlib import Path

from .export import export
from .notes import DetectionSettings
from .pipeline import INSTRUMENTS, TranscribeOptions, transcribe


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="transcriber",
                                 description="Transcribe a music recording to sheet music.")
    ap.add_argument("input", help="audio file (wav, mp3, flac, ogg, ...)")
    ap.add_argument("-o", "--output", action="append",
                    help="output file; .mscz .pdf .musicxml .mxl .mid .png "
                         "(repeatable; default: <input>.mscz)")
    ap.add_argument("--instrument", default="Piano", choices=list(INSTRUMENTS),
                    help="instrument every part is written for (default Piano)")
    ap.add_argument("--max-parts", type=int, default=4,
                    help="most voices/parts to split into (melody ... bass)")
    ap.add_argument("--remove-drums", action="store_true",
                    help="drop the drum stem with Demucs before transcribing (slower)")
    ap.add_argument("--bpm", type=float, help="fixed tempo; tracked (with changes) if omitted")
    ap.add_argument("--time-signature", help="e.g. 3/4 or 6/8; detected if omitted")
    ap.add_argument("--grid", type=int, default=4,
                    help="finest duple subdivision per beat (default 4 = 16ths)")
    ap.add_argument("--no-tuplets", action="store_true", help="disallow triplets/quintuplets")
    ap.add_argument("--no-chords", action="store_true", help="omit chord symbols")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--denoise-strength", type=float, default=0.8)
    ap.add_argument("--onset-threshold", type=float, default=0.5,
                    help="0-1; raise to drop spurious notes, lower to catch soft ones")
    ap.add_argument("--frame-threshold", type=float, default=0.3)
    ap.add_argument("--min-note-ms", type=float, default=50.0)
    ap.add_argument("--title")
    args = ap.parse_args(argv)

    opts = TranscribeOptions(
        denoise=not args.no_denoise,
        denoise_strength=args.denoise_strength,
        detection=DetectionSettings(onset_threshold=args.onset_threshold,
                                    frame_threshold=args.frame_threshold,
                                    min_note_ms=args.min_note_ms),
        instrument=args.instrument,
        max_voices=args.max_parts,
        remove_drums=args.remove_drums,
        bpm=args.bpm,
        time_signature=args.time_signature,
        subdivision=args.grid,
        tuplets=not args.no_tuplets,
        chord_symbols=not args.no_chords,
        title=args.title,
    )
    result = transcribe(args.input, opts, progress=lambda f, d: print(f"[{f:4.0%}] {d}"))
    parts = ", ".join(f"{p.name}: {len(p.quantized)}" for p in result.parts)
    print(f"Parts ({parts}); key {result.key}; {result.bpm:.0f} BPM; "
          f"{len(result.tempo) - 1} tempo change(s)")

    outputs = args.output or [str(Path(args.input).with_suffix(".mscz"))]
    for out in outputs:
        print("Wrote", export(result.score, out, result.notes))


if __name__ == "__main__":
    main()
