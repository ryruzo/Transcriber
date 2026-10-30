"""Key and chord recognition from detected notes, beat by beat.

Chords matter twice: as chord symbols in the score, and as evidence for where
bar lines go, since harmony usually changes on the downbeat.

Each beat gets a pitch-class profile (chroma) from the notes sounding during
it, with the bass weighted up. A hidden Markov model then picks the chord
sequence: emissions are template matches against major/minor/7th/diminished
chords, and transitions encode basic functional harmony (chords tend to
persist; root motion down a fifth, as in V-I or ii-V, is favoured; chords
diatonic to the key are preferred).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import Note

PITCH_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

QUALITIES = {
    "": (0, 4, 7),        # major
    "m": (0, 3, 7),       # minor
    "7": (0, 4, 7, 10),   # dominant seventh
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
}

# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)  # natural + raised 7th (harmonic minor V)


@dataclass(frozen=True)
class Key:
    tonic: int   # pitch class
    minor: bool

    @property
    def uses_flats(self) -> bool:
        # Keys whose signatures use flats (F, Bb, Eb, ... and their relative minors).
        major_tonic = (self.tonic + 3) % 12 if self.minor else self.tonic
        return major_tonic in (5, 10, 3, 8, 1, 6)

    def name(self, pc: int) -> str:
        return (PITCH_NAMES_FLAT if self.uses_flats else PITCH_NAMES_SHARP)[pc % 12]

    @property
    def scale(self) -> set[int]:
        steps = _MINOR_SCALE if self.minor else _MAJOR_SCALE
        return {(self.tonic + s) % 12 for s in steps}

    def __str__(self) -> str:
        return f"{self.name(self.tonic)} {'minor' if self.minor else 'major'}"


@dataclass(frozen=True)
class Chord:
    root: int
    quality: str

    def figure(self, key: Key) -> str:
        return key.name(self.root) + self.quality

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.root + i) % 12 for i in QUALITIES[self.quality])


NO_CHORD = None
CHORDS = [Chord(r, q) for r in range(12) for q in QUALITIES]


def estimate_key(notes: list[Note]) -> Key:
    chroma = np.zeros(12)
    for n in notes:
        chroma[n.pitch % 12] += n.duration * (0.5 + n.amplitude)
    if chroma.sum() == 0:
        return Key(0, False)
    best, best_r = Key(0, False), -2.0
    for tonic in range(12):
        for minor, prof in ((False, _MAJOR_PROFILE), (True, _MINOR_PROFILE)):
            r = float(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1])
            if r > best_r:
                best, best_r = Key(tonic, minor), r
    return best


def beat_chroma(notes: list[Note], beat_pos: np.ndarray, n_beats: int) -> np.ndarray:
    """(n_beats, 12) chroma from notes, given each note's (start, end) in beats.

    `beat_pos` is an (N, 2) array of note start/end positions in beat units.
    The lowest note sounding in each beat gets extra weight, since the bass
    usually carries the chord root.
    """
    chroma = np.zeros((n_beats, 12))
    lowest = np.full(n_beats, 200)
    for n, (s, e) in zip(notes, beat_pos):
        k0, k1 = max(int(np.floor(s)), 0), min(int(np.ceil(e)), n_beats)
        for k in range(k0, k1):
            overlap = min(e, k + 1) - max(s, k)
            if overlap > 0:
                # Notes struck in this beat count more than ones held over.
                fresh = 1.5 if k <= s < k + 1 else 1.0
                chroma[k, n.pitch % 12] += overlap * fresh * (0.5 + n.amplitude)
                lowest[k] = min(lowest[k], n.pitch)
    for k in range(n_beats):
        if lowest[k] < 200:
            chroma[k, lowest[k] % 12] += 0.5 * chroma[k].max()
    return chroma


def _transition_matrix(key: Key, stay: float = 2.0) -> np.ndarray:
    n = len(CHORDS) + 1  # last state = no chord
    t = np.zeros((n, n))
    for i, a in enumerate(CHORDS):
        for j, b in enumerate(CHORDS):
            if i == j:
                t[i, j] = stay
                continue
            motion = (b.root - a.root) % 12
            if motion == 5:      # down a fifth / up a fourth: V-I, ii-V
                t[i, j] = 0.6
            elif motion in (7, 2, 10):
                t[i, j] = 0.3
            if a.root == b.root:  # same root, different colour (C -> C7)
                t[i, j] = 0.8
    return t


def _emissions(chroma: np.ndarray, key: Key) -> np.ndarray:
    n_beats = len(chroma)
    em = np.zeros((n_beats, len(CHORDS) + 1))
    templates = np.zeros((len(CHORDS), 12))
    for i, c in enumerate(CHORDS):
        for j, pc in enumerate(c.pitch_classes):
            templates[i, pc] = 1.0 if j < 3 else 0.6  # 7ths count a little less
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    scale = key.scale
    diatonic = np.array([all(pc in scale for pc in c.pitch_classes) for c in CHORDS])
    # Simple triads first; 7ths and maj7 must earn their extra note.
    simplicity = np.array([0.0 if c.quality in ("", "m") else -0.04 for c in CHORDS])
    norms = np.linalg.norm(chroma, axis=1)
    for k in range(n_beats):
        if norms[k] < 1e-6:
            em[k, :-1] = -10.0
            em[k, -1] = 0.0
            continue
        sim = templates @ (chroma[k] / norms[k])
        em[k, :-1] = 12.0 * (sim + 0.08 * diatonic + simplicity)
        em[k, -1] = 12.0 * 0.35  # "no chord" only wins when nothing fits
    return em


def recognise_chords(chroma: np.ndarray, key: Key) -> list[Chord | None]:
    """Viterbi decoding of one chord per beat."""
    if len(chroma) == 0:
        return []
    em = _emissions(chroma, key)
    trans = _transition_matrix(key)
    n_beats, n_states = em.shape
    score = em[0].copy()
    back = np.zeros((n_beats, n_states), dtype=int)
    for k in range(1, n_beats):
        cand = score[:, None] + trans
        back[k] = np.argmax(cand, axis=0)
        score = cand[back[k], np.arange(n_states)] + em[k]
    path = [int(np.argmax(score))]
    for k in range(n_beats - 1, 0, -1):
        path.append(int(back[k, path[-1]]))
    path.reverse()
    return [CHORDS[s] if s < len(CHORDS) else NO_CHORD for s in path]


def harmonic_change(chords: list[Chord | None]) -> np.ndarray:
    """1.0 where the chord differs from the previous beat's, else 0.

    The first chord after silence doesn't count: music often starts with a
    pickup, and treating its first harmony as a change would pull the bar
    line onto the pickup note.
    """
    hc = np.zeros(len(chords))
    for k in range(1, len(chords)):
        hc[k] = float(chords[k] != chords[k - 1]
                      and chords[k] is not None and chords[k - 1] is not None)
    return hc
