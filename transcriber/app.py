"""Web GUI: paste a YouTube link (or upload audio), get sheet music back.

    python -m transcriber.app
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import gradio as gr

from .export import export, find_musescore
from .notes import DetectionSettings
from .pipeline import INSTRUMENTS, TranscribeOptions, transcribe
from .youtube import download_audio, safe_filename

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
AUTO_METER = "Auto (detect)"


def parse_time(text: str | None) -> float | None:
    """Accept '90', '1:30' or '0:01:30'; blank means not set."""
    text = (text or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+(\.\d+)?(:\d{1,2}(\.\d+)?){0,2}", text):
        raise gr.Error(f"Couldn't read the time '{text}'. Use seconds or m:ss.")
    secs = 0.0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return secs


def _summary(title: str, result, note: str, out_dir: Path) -> str:
    parts = ", ".join(f"{p.name} ({len(p.quantized)} notes)" for p in result.parts)
    meters = []
    for b in result.bars:
        label = f"{3 * b.beats}/8" if result.compound else f"{b.beats}/4"
        if not meters or meters[-1] != label:
            meters.append(label)
    changes = [e for e in result.tempo[1:]]
    tempo = f"♩ = {result.bpm:.0f}"
    if changes:
        tempo += " → " + ", ".join(
            e.kind if e.kind in ("rit.", "accel.") else f"{e.kind} {e.bpm:.0f}"
            for e in changes)
    return (f"### {title}\n"
            f"**Parts:** {parts}  \n"
            f"**Key:** {result.key} · **Meter:** {' → '.join(meters[:6])} · "
            f"**Tempo:** {tempo}  \n"
            f"<small>Rhythm prior: {result.rhythm_prior}. Saved to `{out_dir}`</small>{note}")


def run(url, upload, start, end, instrument, max_parts, remove_drums, bpm, time_sig, grid,
        tuplets, chord_symbols, denoise, sensitivity, progress=gr.Progress()):
    url = (url or "").strip()
    if not url and not upload:
        raise gr.Error("Paste a YouTube link or upload an audio file.")
    start_s, end_s = parse_time(start), parse_time(end)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = OUTPUT_ROOT / stamp
    if url:
        progress(0.02, desc="Downloading audio")
        try:
            dl = download_audio(url, work, start_s, end_s)
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:  # yt-dlp raises many types for bad/blocked links
            raise gr.Error(f"Download failed: {e}")
        audio_path, title = dl.path, dl.title
    else:
        audio_path, title = Path(upload), Path(upload).stem
        if start_s or end_s:
            import soundfile as sf
            import librosa
            y, sr = librosa.load(str(audio_path), sr=None, mono=False)
            a = int((start_s or 0) * sr)
            b = int(end_s * sr) if end_s else y.shape[-1]
            work.mkdir(parents=True, exist_ok=True)
            audio_path = work / "clip.wav"
            sf.write(audio_path, y[..., a:b].T, sr)

    opts = TranscribeOptions(
        denoise=denoise,
        # The slider reads "more notes" to the right, i.e. a lower threshold.
        detection=DetectionSettings(onset_threshold=round(1.0 - sensitivity, 2)),
        instrument=instrument,
        max_voices=int(max_parts),
        remove_drums=remove_drums,
        bpm=bpm or None,
        time_signature=None if time_sig == AUTO_METER else time_sig,
        subdivision={"Eighth notes": 2, "Sixteenth notes": 4, "32nd notes": 8}[grid],
        tuplets=tuplets,
        chord_symbols=chord_symbols,
        title=title,
    )
    try:
        result = transcribe(audio_path, opts,
                            progress=lambda f, d: progress(0.05 + 0.8 * f, desc=d))
    except ValueError as e:
        raise gr.Error(f"{e} Try raising the note sensitivity.")

    progress(0.88, desc="Engraving score")
    out_dir = OUTPUT_ROOT / stamp
    name = safe_filename(title)
    files = [export(result.score, out_dir / f"{name}.musicxml"),
             export(result.score, out_dir / f"{name}.mid", result.notes)]
    pages: list[str] = []
    note = ""
    if find_musescore():
        # A MuseScore failure shouldn't throw away the transcription: the
        # MusicXML and MIDI are already written, so report it and carry on.
        try:
            files.insert(0, export(result.score, out_dir / f"{name}.pdf"))
            files.insert(0, export(result.score, out_dir / f"{name}.mscz"))
            export(result.score, out_dir / f"{name}.png")
            pages = [str(p) for p in sorted(out_dir.glob(f"{name}-*.png"),
                                            key=lambda p: int(p.stem.rsplit("-", 1)[1]))]
        except RuntimeError as e:
            note = (f"\n\n⚠️ {e} The MusicXML file below can still be opened "
                    "in MuseScore or other notation software.")
    else:
        note = ("\n\n⚠️ MuseScore wasn't found, so only MusicXML and MIDI were written. "
                "Install MuseScore 4 (or set `MUSESCORE_PATH`) for .mscz, PDF and previews.")
    return (_summary(title, result, note, out_dir), pages, str(audio_path),
            [str(f) for f in files])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Transcriber") as demo:
        gr.Markdown("# 🎼 Transcriber\nTurn a recording into sheet music. "
                    "Paste a YouTube link or upload an audio file.")
        with gr.Row():
            with gr.Column(scale=2):
                url = gr.Textbox(label="YouTube link",
                                 placeholder="https://www.youtube.com/watch?v=…")
                upload = gr.Audio(label="…or upload audio", type="filepath",
                                  sources=["upload"])
                with gr.Row():
                    start = gr.Textbox(label="Start (optional)", placeholder="0:30")
                    end = gr.Textbox(label="End (optional)", placeholder="1:45")
                with gr.Row():
                    instrument = gr.Dropdown(
                        list(INSTRUMENTS), value="Piano", label="Instrument",
                        info="Every part is written for this instrument")
                    max_parts = gr.Slider(
                        1, 6, value=4, step=1, label="Maximum parts",
                        info="Parts are musical lines (melody, accompaniment, bass), "
                             "split by pitch and voice leading")
                with gr.Accordion("Settings", open=False):
                    with gr.Row():
                        bpm = gr.Number(label="Tempo (BPM, 0 = detect)", value=0,
                                        minimum=0, maximum=300,
                                        info="Setting a tempo disables tempo-change tracking")
                        time_sig = gr.Dropdown(
                            [AUTO_METER, "4/4", "3/4", "2/4", "6/8", "9/8", "12/8", "5/4"],
                            value=AUTO_METER, label="Time signature")
                    with gr.Row():
                        grid = gr.Dropdown(["Eighth notes", "Sixteenth notes", "32nd notes"],
                                           value="Sixteenth notes", label="Shortest note")
                        remove_drums = gr.Checkbox(
                            label="Remove drums first", value=False,
                            info="For band recordings; slower (Demucs)")
                    with gr.Row():
                        tuplets = gr.Checkbox(label="Triplets & other tuplets", value=True)
                        chord_symbols = gr.Checkbox(label="Chord symbols", value=True)
                        denoise = gr.Checkbox(label="Noise reduction", value=True)
                    sensitivity = gr.Slider(0.2, 0.8, value=0.5, step=0.05,
                                            label="Note sensitivity",
                                            info="Higher catches soft notes; "
                                                 "lower removes spurious ones")
                go = gr.Button("Transcribe", variant="primary", size="lg")
            with gr.Column(scale=3):
                summary = gr.Markdown()
                pages = gr.Gallery(label="Score", columns=1, height="auto",
                                   object_fit="contain", show_label=False)
                audio_out = gr.Audio(label="Source audio", interactive=False)
                files = gr.File(label="Downloads", file_count="multiple")

        go.click(run,
                 [url, upload, start, end, instrument, max_parts, remove_drums, bpm,
                  time_sig, grid, tuplets, chord_symbols, denoise, sensitivity],
                 [summary, pages, audio_out, files])
    return demo


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="transcriber.app")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    OUTPUT_ROOT.mkdir(exist_ok=True)
    build_ui().queue().launch(server_port=args.port, inbrowser=not args.no_browser,
                              allowed_paths=[str(OUTPUT_ROOT)])


if __name__ == "__main__":
    main()
