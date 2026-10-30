"""Note detection: neural pitch/onset estimation plus musical clean-up.

The neural network is Spotify's Basic Pitch (Bittner et al., ICASSP 2022): a
small CNN over a harmonic-stacked CQT that outputs per-frame onset, note and
multi-pitch posteriorgrams. It is instrument-agnostic and polyphonic, which
fits a general "recording in, score out" tool far better than hand-tuned
FFT peak-picking (which struggles with overtones and chords).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf


@dataclass
class Note:
    start: float      # seconds
    end: float        # seconds
    pitch: int        # MIDI number
    amplitude: float  # 0-1, model confidence/loudness proxy
    graces: list[int] = field(default_factory=list)  # grace-note pitches before it
    trill: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DetectionSettings:
    onset_threshold: float = 0.5   # higher = fewer, more certain note starts
    frame_threshold: float = 0.3   # higher = notes end sooner / fewer ghost notes
    min_note_ms: float = 50.0    # a 32nd at 120 BPM lasts 62 ms
    min_freq: float | None = 27.5  # A0
    max_freq: float | None = 4186.0  # C8
    # Post-processing
    merge_gap_s: float = 0.5        # max silence bridged when there's no re-attack
    reattack_threshold: float = 0.7  # onset prob. a same-pitch repeat must reach
    min_unattacked_s: float = 0.2    # shorter notes with no onset peak are dropped
    min_relative_amp: float = 0.15  # fast passages are often played softer
    harmonic_ratio: float = 0.7


@dataclass
class OnsetMap:
    """The network's onset posterior, for asking "was this note re-struck?"."""
    times: np.ndarray   # seconds, one per model frame
    onsets: np.ndarray  # (frames, 88) probabilities, column 0 = A0

    def strength(self, t: float, pitch: int, window_s: float = 0.035) -> float:
        col = pitch - 21
        if not 0 <= col < self.onsets.shape[1]:
            return 0.0
        mask = np.abs(self.times - t) <= window_s
        return float(self.onsets[mask, col].max()) if mask.any() else 0.0


def run_model(audio: np.ndarray, sr: int,
              s: DetectionSettings) -> tuple[list[Note], OnsetMap]:
    """Run Basic Pitch on an in-memory signal and return raw note events."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    from basic_pitch.note_creation import model_frames_to_time

    # Basic Pitch reads from disk, so hand it the cleaned audio via a temp WAV.
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, audio, sr)
        output, _, events = predict(
            tmp,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=s.onset_threshold,
            frame_threshold=s.frame_threshold,
            minimum_note_length=s.min_note_ms,
            minimum_frequency=s.min_freq,
            maximum_frequency=s.max_freq,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    finally:
        os.remove(tmp)
    onsets = output["onset"]
    onset_map = OnsetMap(model_frames_to_time(onsets.shape[0]), onsets)
    notes = [Note(float(e[0]), float(e[1]), int(e[2]), float(e[3])) for e in events]
    return notes, onset_map


def merge_fragments(notes: list[Note], onset_map: OnsetMap, max_gap: float,
                    reattack_threshold: float) -> list[Note]:
    """Join a note that the model split in two (same pitch, short gap, no re-attack).

    This happens when a sustained note decays below the frame threshold and is
    picked up again (e.g. when another note's attack excites it). Repeated
    notes also leave gaps, so the deciding evidence is the onset head: a
    genuine repeat shows a strong fresh onset peak, a dropout does not.

    In a fast scale that turns around (C D E D C), the same pitch comes back
    after other notes of the line, so a merge is only allowed if no other
    note in the same register started in between.
    """
    starts = sorted((n.start, n.pitch) for n in notes)

    def line_moved(a: Note, b: Note) -> bool:
        return any(a.start < t < b.start and t_p != a.pitch and abs(t_p - a.pitch) <= 7
                   for t, t_p in starts)

    out: list[Note] = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        prev = out[-1] if out else None
        if (prev and prev.pitch == n.pitch and 0 <= n.start - prev.end <= max_gap
                and onset_map.strength(n.start, n.pitch) < reattack_threshold
                and not line_moved(prev, n)):
            prev.end = max(prev.end, n.end)
        else:
            out.append(Note(n.start, n.end, n.pitch, n.amplitude))
    return sorted(out, key=lambda n: (n.start, n.pitch))


def merge_double_attacks(notes: list[Note], onset_map: OnsetMap,
                         window: float = 0.1) -> list[Note]:
    """One key press reported twice: same pitch, onsets a few frames apart.

    The model sometimes fires twice on one attack. No player repeats a note
    within 100 ms, and the stray extra onset would otherwise force the
    quantizer into odd tuplets, so keep the stronger attack and the union
    of the two durations.
    """
    out: list[Note] = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        prev = out[-1] if out else None
        if prev and prev.pitch == n.pitch and n.start - prev.start < window:
            if onset_map.strength(n.start, n.pitch) > onset_map.strength(prev.start, prev.pitch):
                prev.start = n.start
            prev.end = max(prev.end, n.end)
            prev.amplitude = max(prev.amplitude, n.amplitude)
        else:
            out.append(Note(n.start, n.end, n.pitch, n.amplitude))
    return sorted(out, key=lambda n: (n.start, n.pitch))


def drop_unattacked(notes: list[Note], onset_map: OnsetMap, onset_threshold: float,
                    min_duration: float) -> list[Note]:
    """Drop short notes the onset head never fired for.

    Basic Pitch's "melodia trick" recovers notes from the frame posterior alone,
    which helps with legato lines but also yields brief sub-octave/overtone
    blips. Real notes without a detectable attack are long, so keep those.
    """
    return [n for n in notes
            if n.duration >= min_duration
            or onset_map.strength(n.start, n.pitch) >= onset_threshold]


def drop_ghost_notes(notes: list[Note], min_relative_amp: float) -> list[Note]:
    """Remove notes far quieter than the piece's typical note (bleed, reverb tails)."""
    if not notes:
        return notes
    median = float(np.median([n.amplitude for n in notes]))
    return [n for n in notes if n.amplitude >= min_relative_amp * median]


# Intervals (semitones) at which strong overtones of a lower note land:
# octave, octave+fifth, two octaves.
_HARMONIC_INTERVALS = (12, 19, 24)


def suppress_harmonics(notes: list[Note], ratio: float,
                       onset_tol: float = 0.05) -> list[Note]:
    """Drop notes that are probably overtones of a louder, simultaneous lower note.

    A real doubled octave is usually played about as loudly as its partner, so
    only notes much weaker than the fundamental (amp < ratio * fundamental) go.
    Very low notes an octave *below* a louder simultaneous note are the
    model's sub-octave artifacts and go too.
    """
    keep = []
    for n in notes:
        is_overtone = False
        for m in notes:
            if (n.pitch - m.pitch in _HARMONIC_INTERVALS
                    and abs(n.start - m.start) <= onset_tol
                    and n.amplitude < ratio * m.amplitude):
                overlap = min(n.end, m.end) - max(n.start, m.start)
                if overlap >= 0.7 * n.duration:
                    is_overtone = True
                    break
            if (n.pitch < 40 and m.pitch - n.pitch == 12
                    and abs(n.start - m.start) <= onset_tol
                    and n.amplitude < ratio * m.amplitude):
                is_overtone = True
                break
        if not is_overtone:
            keep.append(n)
    return keep


def drop_isolated_blips(notes: list[Note], max_s: float = 0.12, rel_amp: float = 0.6,
                        window: float = 0.25, register: int = 7) -> list[Note]:
    """Remove short, quiet notes with no neighbours in their register.

    Notes of a fast run are short too, but they always have other notes of
    the line close by; a lone 60 ms blip at half the typical loudness is an
    artifact (a stray overtone or attack transient).
    """
    if not notes:
        return notes
    median = float(np.median([n.amplitude for n in notes]))

    def has_neighbour(n: Note) -> bool:
        return any(o is not n and 0.03 < abs(o.start - n.start) < window
                   and abs(o.pitch - n.pitch) <= register for o in notes)

    return [n for n in notes
            if n.duration >= max_s or n.amplitude >= rel_amp * median or has_neighbour(n)]


def detect_notes(audio: np.ndarray, sr: int,
                 settings: DetectionSettings | None = None) -> list[Note]:
    s = settings or DetectionSettings()
    notes, onset_map = run_model(audio, sr, s)
    notes = merge_double_attacks(notes, onset_map)
    notes = merge_fragments(notes, onset_map, s.merge_gap_s, s.reattack_threshold)
    notes = drop_unattacked(notes, onset_map, s.onset_threshold, s.min_unattacked_s)
    notes = drop_ghost_notes(notes, s.min_relative_amp)
    notes = suppress_harmonics(notes, s.harmonic_ratio)
    notes = drop_isolated_blips(notes)
    return notes
