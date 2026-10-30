"""Fetch audio from YouTube (or any yt-dlp supported site) as WAV."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass
class DownloadedAudio:
    path: Path
    title: str
    duration: float  # seconds, after trimming


def _ffmpeg() -> str:
    # pip-installable static ffmpeg, so users don't need a system install.
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:100] or "audio"


def download_audio(url: str, out_dir: str | Path, start: float | None = None,
                   end: float | None = None, max_duration: float = 20 * 60) -> DownloadedAudio:
    """Download the audio track of `url` to `out_dir` as WAV, optionally trimmed.

    `max_duration` guards against accidentally transcribing hour-long videos;
    it applies to the trimmed section.
    """
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": _ffmpeg(),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        total = float(info.get("duration") or 0)
        lo, hi = start or 0.0, end or total or None
        if hi is not None and hi - lo > max_duration:
            raise ValueError(
                f"Selected section is {(hi - lo) / 60:.0f} min; the limit is "
                f"{max_duration / 60:.0f} min. Set a start/end time to transcribe part of it.")
        ydl.process_info(info)

    wav = out_dir / f"{info['id']}.wav"
    if start or end:
        audio, sr = sf.read(wav, always_2d=False)
        a = int((start or 0) * sr)
        b = int(end * sr) if end else len(audio)
        if b <= a:
            raise ValueError("End time must be after start time.")
        audio = audio[a:b]
        sf.write(wav, audio, sr)
    duration = sf.info(wav).duration
    return DownloadedAudio(wav, info.get("title") or info["id"], duration)
