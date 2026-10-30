"""Optional drum removal before transcription.

Parts are split by voice leading (voices.py), not by instrument sound, but
drums still confuse the pitch model: hits on toms and cymbals come out as
spurious notes. Demucs (htdemucs, Rouard et al. 2023), a hybrid transformer
source-separation network, can remove them; everything except the drum stem
is summed back and transcribed.
"""

from __future__ import annotations

import warnings

import numpy as np
import librosa

MODEL_NAME = "htdemucs"


def without_drums(path: str) -> tuple[np.ndarray, int]:
    """The recording with the drum stem removed (mono, model sample rate)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = get_model(MODEL_NAME)
    model.eval()
    sr = model.samplerate
    audio, _ = librosa.load(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    wav = torch.tensor(audio, dtype=torch.float32)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device="cpu",
                              split=True, overlap=0.25, progress=False,
                              num_workers=0)[0]
    sources = sources * std + mean
    keep = [i for i, name in enumerate(model.sources) if name != "drums"]
    return sources[keep].sum(0).mean(0).numpy().astype(np.float32), sr
