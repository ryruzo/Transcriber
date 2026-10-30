"""Beats, bars and tempo changes.

1. Beat tracking: Beat This! (Foscarin et al., ISMIR 2024), a transformer
   trained on a large multi-genre corpus, gives frame-wise beat and downbeat
   activations. Beats are kept as a list of times, so the tempo can drift
   (rubato, ritardando, accelerando) without notes sliding off the grid.
   Falls back to librosa's dynamic-programming beat tracker if unavailable.
2. Bar lines: a dynamic program over beats chooses where each bar starts and
   how many beats it has (allowing time-signature changes, at a cost). The
   evidence for "this beat is a downbeat" combines the network's downbeat
   activation, chord changes (harmony usually moves on the downbeat), and
   bass onsets.
3. Tempo marks: per-bar tempo is smoothed and segmented into steady stretches
   (metronome marks), gradual changes (rit./accel.) and returns (a tempo).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import librosa


@dataclass
class BeatGrid:
    times: np.ndarray                     # seconds, one per beat
    downbeat_activation: np.ndarray       # per beat, higher = more likely a downbeat
    frame_logits: np.ndarray | None = None  # Beat This! downbeat logits, 50 fps
    beat_logits: np.ndarray | None = None   # Beat This! beat logits, 50 fps

    def with_times(self, times: np.ndarray) -> "BeatGrid":
        return replace(self, times=times,
                       downbeat_activation=_activation(times, self.frame_logits))

    def to_beats(self, t) -> np.ndarray:
        """Map seconds to (fractional) beat index, extrapolating at the ends."""
        t = np.asarray(t, dtype=float)
        idx = np.arange(len(self.times), dtype=float)
        x = np.interp(t, self.times, idx)
        first = self.times[1] - self.times[0]
        last = self.times[-1] - self.times[-2]
        x = np.where(t < self.times[0], (t - self.times[0]) / first, x)
        x = np.where(t > self.times[-1], idx[-1] + (t - self.times[-1]) / last, x)
        return x

    def beat_seconds(self, k: int) -> float:
        k = int(np.clip(k, 0, len(self.times) - 2))
        return float(self.times[k + 1] - self.times[k])


@dataclass
class Bar:
    start: int  # beat index
    beats: int


@dataclass
class TempoEvent:
    beat: int          # beat index where the mark goes
    kind: str          # "tempo", "rit.", "accel.", "a tempo"
    bpm: float


# ---------------------------------------------------------------------------
# Beat tracking

def _beat_this(audio: np.ndarray, sr: int):
    """Beat times, downbeat times, and per-frame beat/downbeat logits (50 fps)."""
    try:
        import torch
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a2f = Audio2Frames(checkpoint_path="final0", device="cpu")
        beat_logits, downbeat_logits = a2f(audio, sr)
        post = Postprocessor(type="minimal")
        beats, downbeats = post(beat_logits, downbeat_logits)
    to_np = (lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
    return (np.asarray(beats), np.asarray(downbeats),
            to_np(beat_logits).astype(float), to_np(downbeat_logits).astype(float))


def _extend(times: np.ndarray, start: float, end: float) -> np.ndarray:
    """Pad the beat list with steady beats so it covers [start, end]."""
    if len(times) < 2:
        times = np.array([0.0, 0.5])
    head = np.median(np.diff(times[:5]))
    tail = np.median(np.diff(times[-5:]))
    pre = []
    t = times[0] - head
    while t >= start - head:
        pre.append(t)
        t -= head
    post = []
    t = times[-1] + tail
    while t <= end + tail:
        post.append(t)
        t += tail
    return np.concatenate([pre[::-1], times, post])


def track_beats(audio: np.ndarray, sr: int, span: tuple[float, float]) -> BeatGrid:
    """Variable-tempo beat grid covering `span` seconds (first/last note)."""
    result = None
    try:
        result = _beat_this(audio, sr)
    except Exception as e:  # model download / runtime problems: fall back
        warnings.warn(f"Beat This! failed ({e}); using librosa beat tracking")
    if result is not None and len(result[0]) >= 4:
        beats, _, b_logits, db_logits = result
        times = _extend(beats, *span)
        return BeatGrid(times, _activation(times, db_logits), db_logits, b_logits)
    _, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    times = _extend(np.asarray(beats), *span)
    return BeatGrid(times, np.zeros(len(times)))


def _activation(times: np.ndarray, logits: np.ndarray | None) -> np.ndarray:
    """Downbeat logit at each beat (max over +-2 frames: peaks sit a frame off)."""
    if logits is None or not len(logits):
        return np.zeros(len(times))
    frames = np.clip(np.round(times * 50).astype(int), 0, len(logits) - 1)
    act = np.array([logits[max(f - 2, 0): f + 3].max() for f in frames])
    outside = (times < 0) | (times > len(logits) / 50)
    act[outside] = np.median(act)
    return act


def _tempo_penalty(bpm: float) -> float:
    """Prefer beats at a readable 60-160 BPM; very fast 'beats' are subdivisions."""
    x = np.log2(bpm / 105.0)
    pen = x * x / (2 * 0.45 ** 2)
    if bpm > 170:
        pen += 2.0
    return float(pen)


def choose_pulse_level(grid: BeatGrid, onset_times: np.ndarray,
                       onset_weights: np.ndarray) -> BeatGrid:
    """Fix the classic beat-tracking octave error.

    Trackers sometimes lock onto the bar (a waltz tracked at 40 BPM) or onto
    eighth notes. We try the tracked pulse x1/2, x1, x2 and x3 and keep the one
    whose beats land on note onsets best, with a prior on readable tempi.
    """
    t = grid.times
    if len(t) < 4 or not len(onset_times):
        return grid
    order = np.argsort(onset_times)
    on, w = onset_times[order], onset_weights[order]

    def support(times: np.ndarray) -> float:
        idx = np.clip(np.searchsorted(on, times), 1, len(on) - 1)
        near = np.minimum(np.abs(on[idx] - times), np.abs(on[idx - 1] - times))
        weight = np.where(np.abs(on[idx] - times) < np.abs(on[idx - 1] - times),
                          w[idx], w[idx - 1])
        inside = (times >= on[0] - 0.1) & (times <= on[-1] + 0.1)
        hit = (near < 0.05) * weight
        return float(hit[inside].mean()) if inside.any() else 0.0

    bpm = 60.0 / float(np.median(np.diff(t)))
    options = {1.0: t}
    for f in (2, 3):
        sub = [t[:-1] + (t[1:] - t[:-1]) * i / f for i in range(f)]
        options[float(f)] = np.sort(np.concatenate(sub + [t[-1:]]))
    # Halving: keep every other beat, on the phase with more downbeat evidence.
    a, b = t[0::2], t[1::2]
    options[0.5] = a if grid.downbeat_activation[0::2].mean() >=         grid.downbeat_activation[1::2].mean() else b

    def score(f: float, times: np.ndarray) -> float:
        new = times if f <= 1 else np.setdiff1d(times, t)
        return 4.0 * support(new) - _tempo_penalty(bpm * f)

    best = max(options, key=lambda f: score(f, options[f]))
    if best == 1.0:
        return grid
    return grid.with_times(options[best])


def refine_beats(grid: BeatGrid, onset_times: np.ndarray, weights: np.ndarray,
                 hop: float = 0.01, smoothness: float = 25.0,
                 tempo_sd: float = 0.35) -> BeatGrid:
    """Variable-tempo beat tracking by dynamic programming (Viterbi).

    The network's beat activation is very good on typical recordings but can
    drop beats in a ritardando, switch metrical level mid-piece (e.g. from
    beats to whole bars), or stumble on sparse textures. Here we re-decode the
    beat sequence from three sources of evidence:

      * the network's beat activation,
      * the onset strength of the transcribed notes (bass weighted up),
      * a prior keeping beats near the corrected global tempo,

    with a penalty on changes between consecutive beat intervals. State =
    (time, interval to previous beat), so the tempo can drift smoothly or
    jump when the evidence demands it.
    """
    t0 = min(onset_times.min(), grid.times[0]) - 0.5
    t1 = max(onset_times.max(), grid.times[-1]) + 0.5
    n = int((t1 - t0) / hop) + 1
    ts = t0 + hop * np.arange(n)

    # Evidence per 10 ms frame.
    onset = np.zeros(n)
    idx = np.clip(np.round((onset_times - t0) / hop).astype(int), 0, n - 1)
    np.add.at(onset, idx, weights)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) * hop / 0.025) ** 2)
    onset = np.convolve(onset, kernel, mode="same")
    scale = np.percentile(onset[onset > 0], 95) if (onset > 0).any() else 1.0
    onset = np.minimum(onset / scale, 1.5)
    neural = np.zeros(n)
    if grid.beat_logits is not None and len(grid.beat_logits):
        frames = np.clip(np.round(ts * 50).astype(int), 0, len(grid.beat_logits) - 1)
        neural = 1 / (1 + np.exp(-grid.beat_logits[frames]))
        neural[(ts < 0) | (ts > len(grid.beat_logits) / 50)] = 0
    # Each beat pays a fixed cost, otherwise summing evidence per beat would
    # always favour more beats (e.g. promoting every eighth note to a beat).
    # An off-beat onset without neural support nets about zero.
    evidence = onset + neural - 1.0

    # Candidate beat intervals (in frames), ~4% apart from 0.2 s to 2 s.
    p0 = float(np.median(np.diff(grid.times)))
    lags = np.unique(np.round(np.geomspace(0.2, 2.0, 60) / hop).astype(int))
    k = len(lags)
    log_l = np.log(lags * hop)
    tempo_cost = 0.5 * ((log_l - np.log(p0)) / tempo_sd) ** 2 * 0.3
    change = smoothness * (log_l[:, None] - log_l[None, :]) ** 2  # [k_new, k_prev]

    score = np.full((n, k), -np.inf)
    back = np.full((n, k), -1, dtype=np.int16)  # -1 = first beat of the chain
    start_frames = int(2.0 / hop)  # the first beat may fall anywhere in the first ~2 s
    for j in range(n):
        prev = j - lags
        valid = prev >= 0
        best = np.full(k, -np.inf)
        if valid.any():
            # rows: new interval k (valid ones), cols: previous interval k'
            cand = score[prev[valid]] - change[valid]
            arg = np.argmax(cand, axis=1)
            best[valid] = cand[np.arange(len(arg)), arg]
            back[j, valid] = arg
        if j < start_frames:
            fresh = best < 0.0
            best[fresh] = 0.0
            back[j, fresh] = -1
        score[j] = evidence[j] - tempo_cost + best

    # End anywhere in the last ~2 s, then follow the back-pointers.
    tail = slice(max(n - int(2.0 / hop), 0), n)
    j = tail.start + int(np.argmax(score[tail].max(axis=1)))
    kk = int(np.argmax(score[j]))
    beats = [j]
    while back[j, kk] >= 0:
        j, kk = j - lags[kk], int(back[j, kk])
        beats.append(j)
    times = ts[np.array(beats[::-1])]
    if len(times) < 4:
        return grid
    return grid.with_times(_extend(times, grid.times[0], grid.times[-1]))


def regularize(grid: BeatGrid, onset_times: np.ndarray, tol: float = 0.07) -> BeatGrid:
    """Re-space beats where no note starts, so held notes don't fake a tempo change.

    During long notes or rests the tracker has nothing to lock onto and
    wanders. Beats with an onset nearby are trusted; runs of unsupported
    beats between them are spread evenly, and those before the first or after
    the last supported beat continue at the neighbouring tempo.
    """
    t = grid.times.copy()
    if len(onset_times) == 0 or len(t) < 4:
        return grid
    on = np.sort(onset_times)
    idx = np.clip(np.searchsorted(on, t), 1, len(on) - 1)
    near = np.minimum(np.abs(on[idx] - t), np.abs(on[idx - 1] - t))
    ok = np.flatnonzero(near < tol)
    if len(ok) < 2:
        return grid
    for a, b in zip(ok, ok[1:]):
        if b - a > 1:
            t[a:b + 1] = np.linspace(t[a], t[b], b - a + 1)
    head = np.median(np.diff(t[ok[0]: ok[0] + 5]))
    tail = np.median(np.diff(t[max(ok[-1] - 4, ok[0]): ok[-1] + 1]))
    t[: ok[0]] = t[ok[0]] - head * np.arange(ok[0], 0, -1)
    t[ok[-1] + 1:] = t[ok[-1]] + tail * np.arange(1, len(t) - ok[-1])
    return grid.with_times(t)


def constant_grid(bpm: float, t0: float, span: tuple[float, float]) -> BeatGrid:
    beat = 60.0 / bpm
    first = t0 - np.ceil((t0 - span[0]) / beat) * beat
    times = np.arange(first, span[1] + 2 * beat, beat)
    return BeatGrid(times, np.zeros(len(times)))


# ---------------------------------------------------------------------------
# Bars and meter

def downbeat_evidence(grid: BeatGrid, harmonic_change: np.ndarray,
                      bass_onsets: np.ndarray) -> np.ndarray:
    """Per-beat score: positive = looks like the first beat of a bar."""
    act = grid.downbeat_activation
    z = (act - act.mean()) / (act.std() + 1e-6) if act.std() > 0 else np.zeros_like(act)
    ev = 1.0 * z + 1.2 * harmonic_change + 0.6 * bass_onsets
    return ev - ev.mean()


def infer_bars(evidence: np.ndarray, meters=(2, 3, 4), change_penalty: float = 2.5,
               bar_cost: float = 0.4) -> list[Bar]:
    """Choose bar starts and lengths with a Viterbi pass over beats.

    States are (beats per bar, position in bar). Every bar start collects that
    beat's downbeat evidence and pays a small `bar_cost` (so 4/4 isn't split
    into 2/4 without reason); switching meter pays `change_penalty`, so a
    time-signature change needs several bars of support.
    """
    n = len(evidence)
    states = [(m, p) for m in meters for p in range(m)]
    index = {s: i for i, s in enumerate(states)}
    prior = {4: 0.6, 3: 0.3, 2: 0.0}
    neg = -1e9
    score = np.full(len(states), neg)
    for (m, p), i in index.items():
        score[i] = prior.get(m, 0.0) + (evidence[0] - bar_cost if p == 0 else 0.0)
    back = np.zeros((n, len(states)), dtype=int)
    for k in range(1, n):
        new = np.full(len(states), neg)
        for (m, p), i in index.items():
            if p > 0:
                j = index[(m, p - 1)]
                best, val = j, score[j]
            else:
                best, val = -1, neg
                for m2 in meters:
                    j = index[(m2, m2 - 1)]
                    v = score[j] - (0.0 if m2 == m else change_penalty)
                    if v > val:
                        best, val = j, v
                val += evidence[k] - bar_cost
            new[i] = val
            back[k, i] = best
        score = new
    # The piece can end mid-bar; take the best final state.
    s = int(np.argmax(score))
    path = [s]
    for k in range(n - 1, 0, -1):
        s = back[k, s]
        path.append(s)
    path.reverse()
    bars: list[Bar] = []
    for k, si in enumerate(path):
        m, p = states[si]
        if p == 0 or k == 0:
            bars.append(Bar(k - p, m))
    return bars


# ---------------------------------------------------------------------------
# Tempo changes

def tempo_events(grid: BeatGrid, bars: list[Bar], beat_unit: float = 1.0,
                 threshold: float = 0.07) -> list[TempoEvent]:
    """Metronome marks, rit./accel. and a tempo from the beat spacing.

    `beat_unit` converts beats/min into quarter notes/min for display
    (1.5 in compound meters where the beat is a dotted quarter).
    """
    if not bars:
        return []
    ibi = np.diff(grid.times)
    bpm_beat = 60.0 / np.maximum(ibi, 1e-3)
    bar_bpm = []
    for b in bars:
        seg = bpm_beat[max(b.start, 0): max(b.start, 0) + b.beats]
        bar_bpm.append(float(np.median(seg)) if len(seg) else bar_bpm[-1] if bar_bpm else 100.0)
    bar_bpm = np.array(bar_bpm)
    # Smooth over three bars to ignore single-bar wobble.
    sm = np.array([np.median(bar_bpm[max(i - 1, 0): i + 2]) for i in range(len(bar_bpm))])

    def bpm_at(i: int) -> float:
        return sm[i] * beat_unit

    first = float(np.median(sm[: min(4, len(sm))]))
    events = [TempoEvent(bars[0].start, "tempo", first * beat_unit)]
    ref, original = first, first
    trend: int | None = None  # +1 accelerating, -1 slowing
    for i in range(1, len(sm)):
        r = sm[i] / ref - 1
        if abs(r) <= threshold:
            trend = None if trend is not None and abs(sm[i] / sm[i - 1] - 1) < 0.03 else trend
            continue
        step = sm[i] / sm[i - 1] - 1
        direction = 1 if r > 0 else -1
        if trend is None and abs(step) > threshold and _steady(sm, i):
            events.append(TempoEvent(bars[i].start, _mark(sm[i], original), bpm_at(i)))
            ref = sm[i]
            continue
        if trend != direction:
            start = i - 1 if abs(sm[i - 1] / ref - 1) > 0.02 and i > 1 else i
            events.append(TempoEvent(bars[start].start,
                                     "accel." if direction > 0 else "rit.", bpm_at(i)))
            trend = direction
        if _steady(sm, i + 1):
            events.append(TempoEvent(bars[i + 1].start, _mark(sm[i + 1], original),
                                     bpm_at(i + 1)))
            ref, trend = sm[i + 1], None
    # Drop duplicates landing on the same bar (keep the later, more specific one).
    dedup: dict[int, TempoEvent] = {}
    for e in events:
        dedup[e.beat] = e
    return sorted(dedup.values(), key=lambda e: e.beat)


def _steady(sm: np.ndarray, i: int, tol: float = 0.03) -> bool:
    return i + 1 < len(sm) and abs(sm[i + 1] / sm[i] - 1) < tol


def _mark(bpm: float, original: float) -> str:
    return "a tempo" if abs(bpm / original - 1) < 0.05 else "tempo"
