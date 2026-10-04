#!/usr/bin/env python3
"""Run the pinned openWakeWord trainer without its obsolete TFLite stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

TRAIN_SOURCE_SHA256 = "a9a994dd10203ef290a251f902e4181d832263876065a6b7c5dfc25f61ca293e"
PIPER_REVISION = "213d4d561ab8a84f71de7dddac827cb07e92c031"
FEATURE_DIGESTS = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
REQUIRED_PACKAGES = {
    "setuptools": "70.3.0",
    "openwakeword": "0.6.0",
    "scipy": "1.13.1",
    "torch": "2.2.2",
    "torchaudio": "2.2.2",
    "torchinfo": "1.8.0",
    "torchmetrics": "1.2.0",
    "speechbrain": "0.5.16",
    "audiomentations": "0.33.0",
    "torch-audiomentations": "0.11.1",
    "acoustics": "0.2.6",
    "mutagen": "1.47.0",
    "pronouncing": "0.2.0",
    "onnx": "1.15.0",
    "PyYAML": "6.0.2",
    "piper-phonemize": "1.1.0",
    "webrtcvad": "2.0.10",
}
TFLITE_CALL = """        convert_onnx_to_tflite(os.path.join(config[\"output_dir\"], config[\"model_name\"] + \".onnx\"),
                               os.path.join(config[\"output_dir\"], config[\"model_name\"] + \".tflite\"))"""
TFLITE_REPLACEMENT = "        logging.info(\"Skipping TFLite export; Oreo deploys ONNX\")"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    training = root / "models/training/openwakeword"
    return {
        "root": root,
        "training": training,
        "config": training / "oreo-training.json",
        "piper": training / "piper-sample-generator",
        "feature_dir": root / "models/cache/openwakeword-v0.5.1-features",
        "negative_test": training
        / "output/oreo/oreo/negative_features_test.npy",
        "negative_validation": training
        / "output/oreo/oreo/negative_validation_features.npy",
        "trained_model": training / "output/oreo/oreo.onnx",
        "cached_model": root / "models/cache/openwakeword-oreo/oreo.onnx",
    }


def train_source() -> tuple[Path, str]:
    spec = importlib.util.find_spec("openwakeword.train")
    if spec is None or spec.origin is None:
        raise RuntimeError("openWakeWord training module is unavailable")
    source_path = Path(spec.origin)
    if sha256(source_path) != TRAIN_SOURCE_SHA256:
        raise RuntimeError("openWakeWord train.py differs from the reviewed 0.6.0 source")
    source = source_path.read_text(encoding="utf-8")
    if source.count(TFLITE_CALL) != 1:
        raise RuntimeError("reviewed TFLite export call was not found exactly once")
    return source_path, source.replace(TFLITE_CALL, TFLITE_REPLACEMENT)


def check_packages() -> None:
    mismatches = []
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+", 1)[0] != expected:
            mismatches.append(f"{package}=={expected} (found {actual})")
    if mismatches:
        raise RuntimeError("training dependencies do not match: " + ", ".join(mismatches))


def check_inputs(project: dict[str, Path], require_generated: bool = True) -> None:
    if require_generated and not project["config"].is_file():
        raise RuntimeError("training config is missing; run the bootstrap script")
    piper = project["piper"]
    if not (piper / "generate_samples.py").is_file():
        raise RuntimeError("pinned Piper sample generator is missing")
    revision = subprocess.run(
        ["git", "-C", str(piper), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PIPER_REVISION:
        raise RuntimeError(f"unexpected Piper revision: {revision}")
    checkpoint = piper / "models/en_US-libritts_r-medium.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size != 204_089_915:
        raise RuntimeError("Piper generator checkpoint is missing or incomplete")
    for name, expected in FEATURE_DIGESTS.items():
        path = project["feature_dir"] / name
        if not path.is_file():
            raise RuntimeError(
                f"cached openWakeWord feature model is missing: {name}; "
                "run scripts/fetch-openwakeword-assets.sh"
            )
        if sha256(path) != expected:
            raise RuntimeError(f"cached openWakeWord feature model is invalid: {name}")


def patch_audio_features(project: dict[str, Path]) -> None:
    import openwakeword.utils as utilities

    original = utilities.AudioFeatures
    feature_dir = project["feature_dir"]

    class CachedAudioFeatures(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.setdefault(
                "melspec_model_path", str(feature_dir / "melspectrogram.onnx")
            )
            kwargs.setdefault(
                "embedding_model_path", str(feature_dir / "embedding_model.onnx")
            )
            kwargs.setdefault("inference_framework", "onnx")
            super().__init__(*args, **kwargs)

    utilities.AudioFeatures = CachedAudioFeatures


def patch_pronunciation_dictionary() -> None:
    import pronouncing

    reviewed = {"orio": ["AO1 R IY0 OW0"]}
    original = pronouncing.phones_for_word

    def phones_for_word(word: str) -> list[str]:
        normalized = word.casefold()
        if normalized in reviewed:
            return reviewed[normalized]
        return original(word)

    pronouncing.phones_for_word = phones_for_word


def check_training_imports(project: dict[str, Path]) -> None:
    patch_pronunciation_dictionary()
    piper_path = str(project["piper"])
    sys.path.insert(0, piper_path)
    try:
        generator = importlib.import_module("generate_samples")
        signature = inspect.signature(generator.generate_samples)
        model = signature.parameters.get("model")
        if model is None or model.default is inspect.Parameter.empty:
            raise RuntimeError("Piper generator does not provide the reviewed default model")
        importlib.import_module("openwakeword.train")
        training_data = importlib.import_module("openwakeword.data")
        adversarial = training_data.generate_adversarial_texts("orio", N=8)
        if len(adversarial) != 8:
            raise RuntimeError("reviewed Orio pronunciation produced no hard negatives")
    finally:
        sys.path.remove(piper_path)


def prepare_validation(project: dict[str, Path]) -> None:
    import numpy as np

    source = project["negative_test"]
    if not source.is_file():
        raise RuntimeError("negative test features are missing; run the augment phase")
    features = np.load(source, mmap_mode="r")
    if features.ndim != 3 or features.shape[2] != 96 or features.shape[0] < 100:
        raise RuntimeError(f"unexpected negative feature shape: {features.shape}")
    output = project["negative_validation"]
    flattened = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(features.shape[0] * features.shape[1], features.shape[2]),
    )
    for start in range(0, features.shape[0], 256):
        batch = features[start : start + 256]
        row = start * features.shape[1]
        flattened[row : row + batch.shape[0] * batch.shape[1]] = batch.reshape(
            -1, features.shape[2]
        )
    flattened.flush()
    print(f"wrote {output} with shape {flattened.shape}")


def run_phase(project: dict[str, Path], phase: str) -> None:
    check_packages()
    check_inputs(project)
    check_training_imports(project)
    source_path, source = train_source()
    if phase == "train" and not project["negative_validation"].is_file():
        raise RuntimeError("validation features are missing; run the augment phase first")
    patch_audio_features(project)
    flag = {
        "generate": "--generate_clips",
        "augment": "--augment_clips",
        "train": "--train_model",
    }[phase]
    previous_argv = sys.argv
    sys.argv = [str(source_path), "--training_config", str(project["config"]), flag]
    try:
        exec(
            compile(source, str(source_path), "exec"),
            {"__name__": "__main__", "__file__": str(source_path)},
        )
    finally:
        sys.argv = previous_argv
    if phase == "augment":
        prepare_validation(project)


def install_candidate(project: dict[str, Path]) -> None:
    check_packages()
    check_inputs(project)
    source = project["trained_model"]
    if not source.is_file():
        raise RuntimeError("trained Oreo ONNX model is missing")
    import onnx

    onnx.checker.check_model(onnx.load(str(source)))
    destination = project["cached_model"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256(destination)
    sidecar = destination.parent.with_suffix(".sha256.local")
    sidecar.write_text(f"{digest}  openwakeword-oreo/oreo.onnx\n", encoding="utf-8")
    print(f"cached candidate at {destination}")
    print(f"SHA-256: {digest}")


def self_test() -> None:
    _, patched = train_source()
    assert TFLITE_CALL not in patched
    assert TFLITE_REPLACEMENT in patched
    assert len(FEATURE_DIGESTS) == 2
    root = paths()["root"]
    recorded = (
        (root / "scripts/fetch-openwakeword-assets.sh").read_text(encoding="utf-8")
        + (root / "models/manifest.toml").read_text(encoding="utf-8")
    )
    assert all(recorded.count(digest) == 2 for digest in FEATURE_DIGESTS.values())
    print("openWakeWord training runner self-test passed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    commands.add_parser("preflight")
    phase = commands.add_parser("phase")
    phase.add_argument("name", choices=("generate", "augment", "train"))
    commands.add_parser("prepare-validation")
    commands.add_parser("install-candidate")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    arguments = parse_arguments()
    project = paths()
    try:
        if arguments.command == "self-test":
            self_test()
        elif arguments.command == "preflight":
            check_packages()
            check_inputs(project)
            check_training_imports(project)
            train_source()
            config = json.loads(project["config"].read_text(encoding="utf-8"))
            if config["target_phrase"] != ["oreo", "orio"]:
                raise RuntimeError("training target is no longer the reviewed identity pair")
            print("openWakeWord ONNX training preflight passed")
        elif arguments.command == "phase":
            run_phase(project, arguments.name)
        elif arguments.command == "prepare-validation":
            prepare_validation(project)
        else:
            install_candidate(project)
        return 0
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
