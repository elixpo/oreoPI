#!/usr/bin/env python3
"""Build Oreo's generic acoustic-candidate openWakeWord training config."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_PHRASES = 128


def corpus_phrases(path: Path) -> tuple[list[str], list[str]]:
    wake: list[str] = []
    ignore: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError("wake intent corpus is malformed")
        split, label, phrase = fields
        if split != "train":
            continue
        if label == "wake":
            wake.append(phrase)
        elif label == "ignore":
            ignore.append(phrase)
        else:
            raise ValueError(f"unsupported corpus label: {label}")
    return wake, ignore


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))


def identity_windows(phrases: list[str], mentions: set[str]) -> list[str]:
    windows: list[str] = []
    for phrase in phrases:
        words = re.findall(r"[a-z0-9]+", phrase.casefold())
        for index, word in enumerate(words):
            if word in mentions:
                windows.append(
                    " ".join(words[max(0, index - 2) : min(len(words), index + 3)])
                )
    return windows


def build_config(base_path: Path, corpus_path: Path, training_root: Path) -> dict[str, object]:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    targets = config.pop("identity_targets", None)
    mentions = config.pop("identity_mentions", None)
    acoustic_negatives = config.pop("acoustic_negative_phrases", None)
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError("identity_targets must be a string list")
    if not isinstance(mentions, list) or not all(isinstance(item, str) for item in mentions):
        raise ValueError("identity_mentions must be a string list")
    if not isinstance(acoustic_negatives, list) or not all(
        isinstance(item, str) for item in acoustic_negatives
    ):
        raise ValueError("acoustic_negative_phrases must be a string list")
    wake, ignore = corpus_phrases(corpus_path)
    mention_words = {word.casefold() for word in mentions}
    carrier_phrases = identity_windows(wake + ignore, mention_words)
    targets = unique(targets + carrier_phrases)
    semantic_negatives = [
        phrase
        for phrase in ignore
        if mention_words.isdisjoint(phrase.casefold().split())
    ]
    negatives = unique(semantic_negatives + acoustic_negatives)
    if not 1 <= len(targets) <= MAX_PHRASES:
        raise ValueError("training requires 1-128 acoustic identity targets")
    if not 5 <= len(negatives) <= MAX_PHRASES:
        raise ValueError("training requires 5-128 acoustic negative phrases")
    config["target_phrase"] = targets
    config["custom_negative_phrases"] = negatives
    root_keys = (
        "piper_sample_generator_path",
        "output_dir",
        "false_positive_validation_data_path",
    )
    for key in root_keys:
        config[key] = str((training_root / str(config[key])).resolve())
    for key in ("rir_paths", "background_paths"):
        config[key] = [
            str((training_root / str(path)).resolve()) for path in config[key]
        ]
    config["feature_data_files"] = {
        name: str((training_root / str(path)).resolve())
        for name, path in config["feature_data_files"].items()
    }
    return config


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="validate corpus-driven config generation")
    build = commands.add_parser("build", help="write the ignored upstream training config")
    build.add_argument(
        "--training-root",
        type=Path,
        default=repo_root / "models/training/openwakeword",
    )
    build.add_argument("--output", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        base = arguments.repo_root / "config/openwakeword-training.base.json"
        corpus = arguments.repo_root / "config/wake-intent-corpus.tsv"
        training_root = (
            arguments.training_root
            if arguments.command == "build"
            else arguments.repo_root / "models/training/openwakeword"
        )
        config = build_config(base, corpus, training_root)
        if arguments.command == "self-test":
            assert len(config["target_phrase"]) >= 60
            assert "oreo" in config["target_phrase"]
            assert "i bought oreo cookies" in config["target_phrase"]
            assert "audio" not in config["target_phrase"]
            assert max(len(phrase.split()) for phrase in config["target_phrase"]) <= 5
            assert "audio" in config["custom_negative_phrases"]
            assert "i bought oreo cookies" not in config["custom_negative_phrases"]
            assert config["feature_data_files"] == {}
            assert config["steps"] == 20_000
            print(
                "openWakeWord config self-test passed: "
                f"{len(config['target_phrase'])} targets, "
                f"{len(config['custom_negative_phrases'])} negatives"
            )
            return 0
        output = arguments.output or training_root / "oreo-training.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
