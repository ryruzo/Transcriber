"""Audio loading and clean-up before transcription."""

from __future__ import annotations

import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

# basic-pitch's model runs at 22050 Hz, so we work there throughout.
SAMPLE_RATE = 22050


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any librosa-readable file as mono float32 at `sr`."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 25.0) -> np.ndarray:
    """Remove DC offset and sub-audible rumble (handling noise, HVAC, mic bumps).

    25 Hz sits below the lowest piano note (A0 = 27.5 Hz), so no notes are lost.
    """
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def remove_hum(audio: np.ndarray, sr: int, max_harmonics: int = 8,
               prominence_db: float = 12.0) -> np.ndarray:
    """Notch out 50/60 Hz mains hum and its harmonics, if present.

    Hum is a steady line spectrum that the pitch model happily reports as a
    held low note (60 Hz is almost exactly B1). Only frequencies that stand
    clearly above their neighbourhood are notched, so recordings without hum
    are untouched.
    """
    freqs, psd = welch(audio, fs=sr, nperseg=min(len(audio), 2 ** 16))
    db = 10 * np.log10(psd + 1e-20)

    def prominence(f: float) -> float:
        peak = db[np.abs(freqs - f) <= 1.0].max()
        ring = (np.abs(freqs - f) > 3) & (np.abs(freqs - f) <= 15)
        return float(peak - np.median(db[ring]))

    out = audio
    for base in (50.0, 60.0):
        if prominence(base) < prominence_db:
            continue
        for k in range(1, max_harmonics + 1):
            f = base * k
            if f >= sr / 2 or (k > 1 and prominence(f) < prominence_db):
                continue
            b, a = iirnotch(f, Q=30 * k ** 0.5, fs=sr)
            out = filtfilt(b, a, out)
    return np.asarray(out, dtype=np.float32)


def _noise_profile(audio: np.ndarray, sr: int, frame_s: float = 0.05,
                   quantile: float = 0.1, min_s: float = 0.5) -> np.ndarray | None:
    """Collect the quietest frames of the recording as a noise sample.

    Most recordings have pauses, breaths or a lead-in where only the noise floor
    is audible; using those frames gives a much better estimate than assuming
    the first N seconds are silent.
    """
    hop = int(frame_s * sr)
    n_frames = len(audio) // hop
    if n_frames < 4:
        return None
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    threshold = np.quantile(rms, quantile)
    quiet = frames[rms <= threshold].ravel()
    if len(quiet) < min_s * sr:
        return None
    return quiet


def denoise(audio: np.ndarray, sr: int, strength: float = 0.8) -> np.ndarray:
    """Spectral-gating noise reduction.

    `strength` (0-1) is how much of the estimated noise floor is removed; keeping
    it below 1 avoids "musical noise" artifacts that the pitch model can mistake
    for notes.
    """
    noise = _noise_profile(audio, sr)
    if noise is not None:
        cleaned = nr.reduce_noise(y=audio, sr=sr, y_noise=noise, stationary=True,
                                  prop_decrease=strength)
    else:
        cleaned = nr.reduce_noise(y=audio, sr=sr, stationary=False,
                                  prop_decrease=strength)
    return cleaned.astype(np.float32)


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(audio))
    return audio if m == 0 else (audio * (peak / m)).astype(np.float32)


def preprocess(path: str, use_denoise: bool = True,
               denoise_strength: float = 0.8) -> np.ndarray:
    """Full clean-up chain: load -> high-pass -> de-hum -> denoise -> normalize.

    Leading silence is deliberately kept: separated stems and the beat
    tracker must share one time origin with the mix.
    """
    audio = load_audio(path)
    audio = highpass(audio, SAMPLE_RATE)
    audio = remove_hum(audio, SAMPLE_RATE)
    if use_denoise:
        audio = denoise(audio, SAMPLE_RATE, denoise_strength)
    return normalize(audio)
