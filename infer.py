#!/usr/bin/env python3
"""
Run inference with a trained voice model on audio files (m4a/mp4/wav/mp3/flac).

Examples:
  python infer.py --model best_model.pt /path/to/sample.m4a
  python infer.py ./clips/ --recursive --show-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from model import VoiceRecognitionSystem

AUDIO_EXTS = {".m4a", ".mp4", ".wav", ".mp3", ".flac"}


def collect_audio_files(paths: Iterable[str], recursive: bool) -> list[Path]:
    files: list[Path] = []

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix.lower() in AUDIO_EXTS:
                files.append(path)
            else:
                raise ValueError(f"Unsupported audio type: {path.suffix}")
        else:
            pattern = "**/*" if recursive else "*"
            for item in path.glob(pattern):
                if item.is_file() and item.suffix.lower() in AUDIO_EXTS:
                    files.append(item)

    return sorted(set(files))


def resolve_default_model() -> Path:
    preferred = [Path("best_model.pt"), Path("voice_model.pt")]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    return Path("voice_model.pt")


def load_system(model_path: Path, device: str | None) -> VoiceRecognitionSystem:
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    num_speakers = checkpoint.get("num_speakers")
    if not num_speakers:
        num_speakers = len(checkpoint.get("speaker_names", [])) or 5

    system = VoiceRecognitionSystem(num_speakers=num_speakers, device=device)
    system.load(str(model_path))
    return system


def format_probs(probs: dict[str, float], top_k: int | None) -> list[tuple[str, float]]:
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run voice model inference on audio files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio files or directories containing audio files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=resolve_default_model(),
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for audio files.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all class probabilities for each file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top-K probabilities (implies --show-all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON lines instead of pretty text.",
    )

    args = parser.parse_args()

    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Provide --model or place best_model.pt in the repo root."
        )

    audio_files = collect_audio_files(args.inputs, args.recursive)
    if not audio_files:
        raise ValueError("No audio files found in the provided inputs.")

    system = load_system(model_path, args.device)

    for audio_path in audio_files:
        speaker, confidence, probs = system.predict(str(audio_path))

        if args.json:
            output = {
                "file": str(audio_path),
                "speaker": speaker,
                "confidence": confidence,
                "probabilities": probs,
            }
            print(json.dumps(output))
            continue

        print(f"{audio_path}: {speaker} ({confidence:.1%})")
        if args.show_all or args.top is not None:
            for name, prob in format_probs(probs, args.top):
                print(f"  - {name}: {prob:.1%}")


if __name__ == "__main__":
    main()
