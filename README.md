# Transcriber

Turns audio recordings of music into sheet music (`.mscz`, `.pdf`, MusicXML, MIDI).

## Pipeline

```
mix ─► (optional drum removal) ─► clean-up ─► Basic Pitch ─► ornaments
 │                                                        │
 └─► Beat This! + DP beats ─► chords + bass ─► bars/meter ─┤
                                                          ▼
     rhythm-prior quantization ─► voice separation ─► note lengths ─► score
```

1. **Clean-up** (`audio.py`): 25 Hz high-pass, 50/60 Hz hum notch (only if hum is
   present), spectral-gating denoise using the quietest frames as the noise profile.
2. **Drum removal (optional)** (`separation.py`): [Demucs](https://github.com/facebookresearch/demucs)
   removes the drum stem so drum hits don't become spurious notes. Parts are *not* split
   by instrument sound.
3. **Note detection** (`notes.py`): [Basic Pitch](https://github.com/spotify/basic-pitch)
   (polyphonic CNN, via ONNX; notes down to 50 ms so 32nd-note runs survive), then:
   double attacks merged, fragments merged only when there's no re-attack *and* the line
   didn't move in between (so turning scales keep their repeated pitches), overtones and
   sub-octave artifacts removed, and short quiet blips with no neighbours dropped.
4. **Ornaments** (`ornaments.py`): fast two-pitch alternations become one trilled note;
   very short notes leading into a longer neighbour (< 1/5 beat ahead) become grace notes,
   so neither shows up as fake 32nd/64th rhythms.
5. **Beats, tempo, meter** (`beats.py`):
   - [Beat This!](https://github.com/CPJKU/beat_this) (ISMIR 2024 transformer) gives beat and
     downbeat activations;
   - the pulse level is corrected (trackers often lock onto bars or eighths);
   - a Viterbi beat tracker re-decodes beats from the neural activation + transcribed onsets
     with a tempo-smoothness cost, so ritardando/accelerando and sudden changes are followed;
   - bar lines come from a second Viterbi pass over (beats-per-bar, position) states, using
     the neural downbeat activation, **chord changes** and **bass onsets** as evidence, with
     a cost for time-signature changes (so 4/4 → 3/4 is found, single-bar noise isn't);
   - per-bar tempo is segmented into metronome marks, *rit.*, *accel.* and *a tempo*.
6. **Harmony** (`harmony.py`): key by Krumhansl–Kessler profiles; one chord per beat from an
   HMM over major/minor/7th/maj7/dim chords, with transitions favouring persistence,
   falling-fifth root motion (V–I, ii–V) and diatonic chords. Used for bar lines and as
   chord symbols in the score.
7. **Rhythm** (`rhythm.py`, `rhythm_prior.py`): onsets are placed on the variable-tempo beat
   grid, then each beat's onsets are explained by the most probable *written* pattern:
   timing error (Gaussian) − log P(pattern | family) − log P(family | previous family).
   The prior is **learned from 15k public-domain MuseScore scores** (PDMX), so a clean
   triplet beats a weird 64th figure. On top of it, tuplet beats pay an extra simplicity
   cost and quintuplets are off by default, so tuplets only appear when the timing clearly
   calls for them. Beats holding more notes than the grid allows (fast runs) get the next
   finer grid instead of being squashed into chords; near-simultaneous onsets are only
   grouped within a tempo-scaled window. Music that is mostly triplets is written in 6/8,
   9/8 or 12/8 instead.
8. **Voices = parts** (`voices.py`): parts are musical lines split by pitch and voice
   leading, all written for piano by default. Notes struck together with the same length
   form a chord (unless more than an octave apart); everything else is matched onset by
   onset to the running voice it continues best (small leaps, staying in register, no
   crossing a sounding voice, no cutting off a ringing note, a cost for new voices). So a
   held whole-note bass line is one part, the moving melody another, accompaniment a
   third. Sparse voices merge into their neighbours (max 4 parts by default).
9. **Note lengths** (`voices.py`): within a voice, a note that sounds into or nearly up to
   the next note lasts until it; otherwise it ends on the nearest simple value with a
   simple rest after it. Notes aren't stretched across a bar line once their sound has
   died, chord tones share one length, and every end sits on its beat's grid (thirds in a
   triplet beat), so ties only appear where a note really crosses a bar line.
10. **Notation** (`score.py`): one staff per voice, named by role (Melody, Voice 2…, Bass),
   a real **pickup measure** when the music starts before the first downbeat, measures with
   time-signature changes, tempo marks, chord symbols, grace notes, trills.
   **Safety** (`safety.py`): every measure is checked (fills the bar exactly, every value
   writable, tuplets complete within one beat, plain notes on plain positions); a failing
   measure is rebuilt on a plain grid, and if MuseScore still refuses a file, measures
   with tuplets are simplified and the export retried.
11. **Export** (`export.py`): MusicXML via music21; `.mscz`/`.pdf`/`.png` through the
   MuseScore CLI; `.mid` is the unquantized performance, one track per part.

## Setup (Windows, Python 3.12)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install basic-pitch==0.4.0 --no-deps
.venv/Scripts/python -m pip install -r requirements.txt
```

Model weights for Demucs and Beat This! download automatically on first use.

MuseScore 4 is needed for `.mscz`/`.pdf`; it's found automatically in the default
install location, or set `MUSESCORE_PATH`.

## Usage

### GUI

```bash
.venv/Scripts/python -m transcriber.app
```

Opens a local web page (http://localhost:7860). Paste a YouTube link (any site
[yt-dlp](https://github.com/yt-dlp/yt-dlp) supports works) or upload an audio file,
optionally set a start/end time (`1:30` or `90`), and press **Transcribe**. You get a
score preview, the source audio, and `.mscz` / `.pdf` / `.musicxml` / `.mid` downloads.
Results are also saved under `outputs/<timestamp>/`. Sections longer than 20 minutes
are refused; trim with start/end. ffmpeg comes bundled via `imageio-ffmpeg`.

### Command line

```bash
.venv/Scripts/python -m transcriber recording.wav -o score.mscz -o score.pdf
```

Useful options: `--max-parts 3` (voices to split into), `--instrument Violin` (every part
is piano unless set), `--remove-drums` (Demucs pass for band recordings), `--bpm 96`
(fixed tempo; disables tempo-change tracking), `--time-signature 3/4` (otherwise
detected, including changes), `--no-tuplets`, `--no-chords`, `--onset-threshold 0.6`
(fewer spurious notes), `--no-denoise`. Run with `-h` for all.

## Evaluating accuracy

```bash
.venv/Scripts/python scripts/evaluate.py --snr 30 10      # note detection (pitch + onset F1)
.venv/Scripts/python scripts/evaluate_rhythm.py           # notation: rhythm, meter, tempo, ornaments
```

`evaluate_rhythm.py` renders pieces with known notation and voices (triplets, a waltz, a
ritardando, 6/8, grace notes + trill, a 4/4→3/4 change, a sudden tempo change, 16th/32nd
runs, syncopation, a pickup) and checks rhythm, note lengths, voice assignment, ties per
note, fake tuplets, pickup measures, meter, bar lines, tempo marks and ornaments. Current results: voices 100% correct on all pieces, ~0 ties per note; rhythm,
meter and bar lines exact on all but the sudden tempo-jump piece (see limitations).

## Retraining the rhythm prior

```bash
# PDMX MIDI exports (214 MB) + subset lists from https://zenodo.org/records/15571083
.venv/Scripts/python scripts/train_rhythm_prior.py data/pdmx/mid     --subset data/pdmx/subset_paths/deduplicated.txt --max-files 15000
```

## Known limitations

- A sudden large tempo jump where the new beat isn't accented can be read at half speed.
- Voices are separated by pitch and voice leading only, so two instruments playing in the
  same register in the same rhythm end up in one part.
- Note durations on decaying instruments are inferred (the release can't be heard).
