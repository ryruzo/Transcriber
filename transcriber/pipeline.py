"""End-to-end: audio file -> notes -> beats/bars/chords -> voices -> score.

    mix -> (optional drum removal) -> clean-up -> Basic Pitch -> ornaments
     |                                                          |
     +-> Beat This! + DP beats -> chords + bass -> bars/meter --+
                                                                v
        rhythm-prior quantization -> voice separation -> note lengths -> score
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa
from music21 import stream

from .audio import SAMPLE_RATE, denoise, highpass, normalize, preprocess, remove_hum
from .beats import Bar, BeatGrid, TempoEvent, choose_pulse_level, constant_grid, \
    downbeat_evidence, infer_bars, refine_beats, regularize, tempo_events, track_beats
from .harmony import Chord, Key, beat_chroma, estimate_key, harmonic_change, \
    recognise_chords
from .notes import DetectionSettings, Note, detect_notes
from .ornaments import detect_ornaments
from .rhythm import QuantizedNote, RhythmOptions, fit_phase, is_compound, quantize_part
from .rhythm_prior import load_prior
from .score import PartSpec, ScoreLayout, build_score
from .voices import resolve_lengths, role_names, separate_voices

# Display name -> music21 instrument class.
INSTRUMENTS = {
    "Piano": "Piano",
    "Organ": "Organ",
    "Guitar": "AcousticGuitar",
    "Violin": "Violin",
    "Viola": "Viola",
    "Cello": "Violoncello",
    "Flute": "Flute",
    "Clarinet": "Clarinet",
    "Trumpet": "Trumpet",
    "Voice": "Vocalist",
}
# Instruments whose notes fade on their own, so the release can't be heard.
_DECAYING = {"Piano", "AcousticGuitar"}


@dataclass
class TranscribeOptions:
    denoise: bool = True
    denoise_strength: float = 0.8
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    remove_drums: bool = False        # Demucs pass to drop the drum stem first
    instrument: str = "Piano"         # every part is written for this instrument
    max_voices: int = 4               # parts are voices: melody ... bass
    bpm: float | None = None          # None = track beats (variable tempo)
    time_signature: str | None = None  # None = detect, e.g. "3/4", "6/8"
    subdivision: int = 4              # finest duple subdivision per beat
    tuplets: bool = True              # triplets, sextuplets, quintuplets
    chord_symbols: bool = True
    title: str | None = None


@dataclass
class PartResult:
    name: str
    instrument: str
    quantized: list[QuantizedNote]


@dataclass
class Transcription:
    notes: list[Note]                 # as heard (seconds), for MIDI export
    parts: list[PartResult]           # one per voice, top to bottom
    key: Key
    bars: list[Bar]
    tempo: list[TempoEvent]
    chords: list[Chord | None]
    compound: bool
    score: stream.Score
    rhythm_prior: str

    @property
    def bpm(self) -> float:
        return self.tempo[0].bpm if self.tempo else 0.0


def _parse_time_signature(ts: str) -> tuple[tuple[int, ...], bool]:
    num, den = (int(x) for x in ts.split("/"))
    if den == 8 and num % 3 == 0:
        return (num // 3,), True
    return (num,), False


def _detect(path: str, mix: np.ndarray, o: TranscribeOptions, progress) -> list[Note]:
    audio = mix
    if o.remove_drums:
        try:
            from .separation import without_drums
            progress(0.15, "Removing drums")
            stem, sr = without_drums(path)
            audio = librosa.resample(stem, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio = remove_hum(highpass(audio, SAMPLE_RATE), SAMPLE_RATE)
            if o.denoise:
                audio = denoise(audio, SAMPLE_RATE, o.denoise_strength)
            audio = normalize(audio)
        except Exception as e:
            warnings.warn(f"Drum removal failed ({e}); transcribing the full mix")
    progress(0.4, "Detecting notes")
    return detect_notes(audio, SAMPLE_RATE, o.detection)


def _bass_onsets(notes: list[Note], grid: BeatGrid) -> np.ndarray:
    """1.0 for beats where a low note starts right on the beat."""
    out = np.zeros(len(grid.times))
    for n in notes:
        if n.pitch < 55:
            x = float(grid.to_beats(n.start))
            k = int(round(x))
            if 0 <= k < len(out) and abs(x - k) < 0.12:
                out[k] = 1.0
    return out


def transcribe(path: str | Path, opts: TranscribeOptions | None = None,
               progress=None) -> Transcription:
    o = opts or TranscribeOptions()
    progress = progress or (lambda frac, desc: None)
    path = str(path)

    progress(0.05, "Cleaning audio")
    mix = preprocess(path, o.denoise, o.denoise_strength)
    notes = _detect(path, mix, o, progress)
    if not notes:
        raise ValueError("No notes were detected.")
    span = (min(n.start for n in notes), max(n.end for n in notes))

    progress(0.6, "Tracking beats")
    onsets = np.array([n.start for n in notes])
    if o.bpm:
        grid = constant_grid(o.bpm, fit_phase(onsets, o.bpm), span)
    else:
        grid = track_beats(mix, SAMPLE_RATE, span)
        grid = choose_pulse_level(grid, onsets, np.array([0.3 + n.amplitude for n in notes]))
        weights = np.array([0.3 + n.amplitude + (0.5 if n.pitch < 55 else 0.0)
                            for n in notes])
        grid = regularize(refine_beats(grid, onsets, weights), onsets)

    def beat_seconds(t: float) -> float:
        return grid.beat_seconds(int(grid.to_beats(t)))

    notes = detect_ornaments(notes, beat_seconds)

    progress(0.7, "Finding chords and bar lines")
    key = estimate_key(notes)
    pos = np.array([[grid.to_beats(n.start), grid.to_beats(n.end)] for n in notes])
    chroma = beat_chroma(notes, pos, len(grid.times))
    chords = recognise_chords(chroma, key)
    evidence = downbeat_evidence(grid, harmonic_change(chords), _bass_onsets(notes, grid))
    if o.time_signature:
        meters, compound = _parse_time_signature(o.time_signature)
    else:
        meters, compound = (2, 3, 4), False
    bars = infer_bars(evidence, meters)

    progress(0.78, "Quantizing rhythms")
    prior = load_prior()
    quantized, families = quantize_part(notes, grid, RhythmOptions(o.subdivision, o.tuplets),
                                        prior)
    if not o.time_signature:
        # Mostly triplet beats means the music is really in 6/8, 9/8 or 12/8.
        compound = is_compound(families)

    progress(0.84, "Separating voices")
    inst = INSTRUMENTS.get(o.instrument, "Piano")
    voices = separate_voices(quantized, o.max_voices)
    for v in voices:
        resolve_lengths(v, families, [b.start for b in bars], decaying=inst in _DECAYING)
    names = role_names(len(voices))
    parts = [PartResult(name, inst, v.notes) for name, v in zip(names, voices)]

    tempo = tempo_events(grid, bars)
    changes = []
    if o.chord_symbols:
        prev = None
        for k, ch in enumerate(chords):
            if ch is not None and ch != prev:
                changes.append((k, ch))
            prev = ch if ch is not None else prev

    progress(0.9, "Engraving")
    specs = [PartSpec(p.name, p.instrument, p.quantized, grand_staff=False) for p in parts]
    layout = ScoreLayout(bars, compound, tempo, changes)
    score = build_score(specs, layout, key, title=o.title or Path(path).stem)
    return Transcription(notes, parts, key, bars, tempo, chords, compound, score, prior.source)
