# Voice Model Inference

This repo contains a lightweight voice recognition model trained with `train.py`. Use `infer.py` to run inference on individual audio files (including `.m4a`).

## Quick start

1) Install deps (torchaudio + torch + numpy).
2) Make sure `best_model.pt` or `voice_model.pt` exists in the repo root.

## Run inference

- Single file:

  - `python infer.py --model best_model.pt /path/to/sample.m4a`

- Directory (recursive):

  - `python infer.py ./clips --recursive --show-all`

## Notes

- Supported audio formats: `.m4a`, `.mp4`, `.wav`, `.mp3`, `.flac`.
- The default input window uses the **full file duration** (variable length). During training, inputs are padded to the longest clip in your dataset.
- If you change `DURATION_SEC` in `train.py`, you must retrain the model.
- If `.m4a` fails to load, ensure your torchaudio build has FFmpeg support.

## Train (reference)

- `python train.py`
