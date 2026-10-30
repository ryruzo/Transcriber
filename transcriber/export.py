"""Write scores to MusicXML, MIDI, and (via MuseScore) .mscz / .pdf."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pretty_midi
from music21 import stream

from .notes import Note

_MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]


def find_musescore() -> str | None:
    env = os.environ.get("MUSESCORE_PATH")
    if env and Path(env).exists():
        return env
    for name in ("MuseScore4", "mscore4portable", "mscore", "musescore"):
        found = shutil.which(name)
        if found:
            return found
    return next((c for c in _MUSESCORE_CANDIDATES if Path(c).exists()), None)


# General MIDI programs for the part names we produce.
_GM_PROGRAMS = {"Piano": 0, "Organ": 19, "Guitar": 24, "Bass": 33, "Violin": 40,
                "Viola": 41, "Cello": 42, "Trumpet": 56, "Clarinet": 71,
                "Flute": 73, "Voice": 52}


def write_performance_midi(notes: list[Note] | dict[str, list[Note]],
                           path: str | Path) -> None:
    """Unquantized MIDI (exactly what was heard), one track per part."""
    parts = notes if isinstance(notes, dict) else {"Piano": notes}
    pm = pretty_midi.PrettyMIDI()
    for name, part in parts.items():
        program = _GM_PROGRAMS.get(name.split(" ")[0], 0)
        inst = pretty_midi.Instrument(program=program, name=name)
        for n in part:
            vel = int(max(1, min(127, round(40 + 87 * n.amplitude))))
            inst.notes.append(pretty_midi.Note(vel, n.pitch, n.start, n.end))
        pm.instruments.append(inst)
    pm.write(str(path))


def musescore_convert(src: Path, dst: Path) -> None:
    exe = find_musescore()
    if exe is None:
        raise RuntimeError(
            "MuseScore not found; install it or set MUSESCORE_PATH to export "
            f"{dst.suffix} files (MusicXML export works without it).")
    # Image formats are written one file per page: name-1.png, name-2.png, ...
    paged = dst.with_name(f"{dst.stem}-1{dst.suffix}")
    # Remove stale output first, so a failed run can't pass for a fresh one.
    for old in [dst, *dst.parent.glob(f"{dst.stem}-*{dst.suffix}")]:
        old.unlink(missing_ok=True)
    for _ in range(2):  # MuseScore's CLI occasionally crashes on start-up; retry once
        result = subprocess.run([exe, "-o", str(dst), str(src)],
                                capture_output=True, text=True, timeout=300)
        if dst.exists() or paged.exists():
            break
    if not (dst.exists() or paged.exists()):
        raise RuntimeError(f"MuseScore couldn't convert the score to {dst.suffix} "
                           f"(exit code {result.returncode}). {result.stderr[-2000:]}")


def export(score: stream.Score, path: str | Path,
           notes: list[Note] | dict[str, list[Note]] | None = None) -> Path:
    """Write `score` to `path`; the format comes from the file extension."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in (".musicxml", ".xml", ".mxl"):
        score.write("mxl" if ext == ".mxl" else "musicxml", fp=str(path))
    elif ext in (".mid", ".midi"):
        if notes is not None:
            write_performance_midi(notes, path)
        else:
            score.write("midi", fp=str(path))
    elif ext in (".mscz", ".mscx", ".pdf", ".png", ".svg"):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / (path.stem + ".musicxml")
            score.write("musicxml", fp=str(xml))
            try:
                musescore_convert(xml, path)
            except RuntimeError:
                # MuseScore still refused it: simplify every measure that has
                # tuplets to a plain grid and try once more.
                from .safety import ensure_valid
                if ensure_valid(score, getattr(score, "beat_ql", 1), force=True) == 0:
                    raise
                score.write("musicxml", fp=str(xml))
                musescore_convert(xml, path)
    else:
        raise ValueError(f"Unsupported output format: {ext}")
    return path
