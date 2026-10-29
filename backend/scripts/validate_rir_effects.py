"""Measure paired feature changes from known simulated room impulse responses."""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.services.audio_processor import AudioProcessor
from scripts.prepare_data import load_and_apply_rir

ROOM_TYPES = ("smallroom", "mediumroom", "largeroom")
LABELS = ("bonafide", "spoof")
REQUIRED_COLUMNS = {"split", "label", "utterance_id", "audio_path"}
DELTA_FEATURES = (
    "spectral_centroid_delta_hz",
    "spectral_rolloff_delta_hz",
    "rms_change_db",
    "decay_proxy_delta_seconds",
    "spectrogram_mean_absolute_difference_db",
    "duration_delta_seconds",
)
ROOM_RECOGNITION_FEATURES = (
    "spectral_centroid_delta_hz",
    "spectral_rolloff_delta_hz",
    "rms_change_db",
    "decay_proxy_delta_seconds",
    "spectrogram_mean_absolute_difference_db",
)
CSV_FIELDS = [
    "utterance_id",
    "label",
    "room_type",
    "room_id",
    "rir_id",
    "dry_duration_seconds",
    "wet_duration_seconds",
    *DELTA_FEATURES,
]


def _sample_training_rows(
    manifest_path: Path,
    samples_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    rows_by_label: dict[str, list[dict[str, str]]] = {
        label: [] for label in LABELS
    }
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        reader = csv.DictReader(manifest)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            label = row["label"].strip().lower()
            if row["split"] == "train" and label in rows_by_label:
                rows_by_label[label].append(row)

    insufficient = {
        label: len(rows)
        for label, rows in rows_by_label.items()
        if len(rows) < samples_per_label
    }
    if insufficient:
        raise ValueError(
            "Not enough train rows per label: "
            + ", ".join(f"{label}={count}" for label, count in insufficient.items())
        )

    generator = random.Random(seed)
    return [
        row
        for label in LABELS
        for row in generator.sample(rows_by_label[label], samples_per_label)
    ]


def _select_rirs(
    rir_root: Path,
    rirs_per_room_type: int,
    seed: int,
) -> list[tuple[str, Path]]:
    generator = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for room_type in ROOM_TYPES:
        room_type_path = rir_root / room_type
        if not room_type_path.is_dir():
            raise FileNotFoundError(f"RIR room type not found: {room_type_path}")
        rooms = sorted(
            room
            for room in room_type_path.iterdir()
            if room.is_dir() and any(room.glob("*.wav"))
        )
        if len(rooms) < rirs_per_room_type:
            raise ValueError(
                f"Expected {rirs_per_room_type} rooms with RIR WAV files under "
                f"{room_type_path}, found {len(rooms)}."
            )

        for room in generator.sample(rooms, rirs_per_room_type):
            rir_files = sorted(room.glob("*.wav"))
            rir_path = generator.choice(rir_files)
            selected.append((room_type, rir_path))
    return selected


def _extract_features(
    processor: AudioProcessor,
    audio: np.ndarray,
) -> dict[str, Any]:
    result = processor.extract_features_from_audio(audio, processor.target_sr)
    return {
        "duration_seconds": result["metadata"]["duration_seconds"],
        "mean_spectral_centroid_hz": result["acoustic_features"][
            "mean_spectral_centroid_hz"
        ],
        "mean_spectral_rolloff_hz": result["acoustic_features"][
            "mean_spectral_rolloff_hz"
        ],
        "mean_rms_energy": result["acoustic_features"]["mean_rms_energy"],
        "reverb_decay_proxy_seconds": result["acoustic_features"][
            "reverb_decay_proxy_seconds"
        ],
        "spectrogram_values_db": np.asarray(
            result["spectrogram"]["values_db"],
            dtype=np.float64,
        ),
    }


def _paired_deltas(
    dry: dict[str, Any],
    wet: dict[str, Any],
) -> dict[str, float]:
    dry_rms = float(dry["mean_rms_energy"])
    wet_rms = float(wet["mean_rms_energy"])
    rms_change_db = (
        20.0 * np.log10(wet_rms / dry_rms)
        if dry_rms > np.finfo(np.float64).eps
        and wet_rms > np.finfo(np.float64).eps
        else 0.0
    )
    return {
        "spectral_centroid_delta_hz": (
            float(wet["mean_spectral_centroid_hz"])
            - float(dry["mean_spectral_centroid_hz"])
        ),
        "spectral_rolloff_delta_hz": (
            float(wet["mean_spectral_rolloff_hz"])
            - float(dry["mean_spectral_rolloff_hz"])
        ),
        "rms_change_db": float(rms_change_db),
        "decay_proxy_delta_seconds": (
            float(wet["reverb_decay_proxy_seconds"])
            - float(dry["reverb_decay_proxy_seconds"])
        ),
        "spectrogram_mean_absolute_difference_db": float(
            np.mean(
                np.abs(
                    wet["spectrogram_values_db"] - dry["spectrogram_values_db"]
                )
            )
        ),
        "duration_delta_seconds": (
            float(wet["duration_seconds"]) - float(dry["duration_seconds"])
        ),
    }


def _room_type_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_room_type: dict[str, list[dict[str, Any]]] = {
        room_type: [] for room_type in ROOM_TYPES
    }
    for row in rows:
        by_room_type[row["room_type"]].append(row)

    room_summary: dict[str, Any] = {}
    for room_type, room_rows in by_room_type.items():
        if not room_rows:
            raise ValueError(f"No paired examples found for room type {room_type}.")
        room_summary[room_type] = {
            "pairs": len(room_rows),
            "unique_rooms": len({row["room_id"] for row in room_rows}),
            "mean_feature_deltas": {
                feature: float(
                    np.mean([row[feature] for row in room_rows])
                )
                for feature in DELTA_FEATURES
            },
            "mean_absolute_feature_deltas": {
                feature: float(
                    np.mean([abs(row[feature]) for row in room_rows])
                )
                for feature in DELTA_FEATURES
            },
        }
    return room_summary


def _recognize_room_type(
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    source_labels: dict[str, str] = {}
    for row in rows:
        source_id = row["utterance_id"]
        label = row["label"]
        if source_id in source_labels and source_labels[source_id] != label:
            raise ValueError(f"Conflicting labels for source {source_id}.")
        source_labels[source_id] = label

    sources_by_label: dict[str, list[str]] = {label: [] for label in LABELS}
    for source_id, label in source_labels.items():
        if label in sources_by_label:
            sources_by_label[label].append(source_id)

    generator = random.Random(seed)
    held_out_sources: set[str] = set()
    for label, source_ids in sources_by_label.items():
        if len(source_ids) < 2:
            raise ValueError(
                "Need at least two unique source recordings per label to "
                "evaluate room recognition without source leakage."
            )
        generator.shuffle(source_ids)
        held_out_count = max(1, round(len(source_ids) * 0.25))
        held_out_sources.update(source_ids[:held_out_count])

    rooms_by_type: dict[str, list[str]] = {
        room_type: [] for room_type in ROOM_TYPES
    }
    for row in rows:
        if row["room_id"] not in rooms_by_type[row["room_type"]]:
            rooms_by_type[row["room_type"]].append(row["room_id"])

    held_out_rooms: set[str] = set()
    for room_type, room_ids in rooms_by_type.items():
        if len(room_ids) < 2:
            raise ValueError(
                "Need at least two unique rooms per room type to evaluate "
                "recognition on unseen rooms."
            )
        generator.shuffle(room_ids)
        held_out_count = max(1, round(len(room_ids) * 0.25))
        held_out_rooms.update(room_ids[:held_out_count])

    training_rows = [
        row
        for row in rows
        if row["utterance_id"] not in held_out_sources
        and row["room_id"] not in held_out_rooms
    ]
    test_rows = [
        row
        for row in rows
        if row["utterance_id"] in held_out_sources
        and row["room_id"] in held_out_rooms
    ]
    if not training_rows or not test_rows:
        raise ValueError("Could not create source-disjoint train and test groups.")

    training_matrix = np.asarray(
        [
            [row[feature] for feature in ROOM_RECOGNITION_FEATURES]
            for row in training_rows
        ],
        dtype=np.float64,
    )
    mean = np.mean(training_matrix, axis=0)
    scale = np.std(training_matrix, axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0

    centroids = {
        room_type: np.mean(
            [
                (
                    np.asarray(
                        [
                            row[feature]
                            for feature in ROOM_RECOGNITION_FEATURES
                        ]
                    )
                    - mean
                )
                / scale
                for row in training_rows
                if row["room_type"] == room_type
            ],
            axis=0,
        )
        for room_type in ROOM_TYPES
    }
    confusion = {
        actual: {predicted: 0 for predicted in ROOM_TYPES}
        for actual in ROOM_TYPES
    }
    correct = 0
    for row in test_rows:
        vector = (
            np.asarray(
                [row[feature] for feature in ROOM_RECOGNITION_FEATURES],
                dtype=np.float64,
            )
            - mean
        ) / scale
        predicted = min(
            ROOM_TYPES,
            key=lambda room_type: float(
                np.linalg.norm(vector - centroids[room_type])
            ),
        )
        confusion[row["room_type"]][predicted] += 1
        correct += int(predicted == row["room_type"])

    return {
        "method": (
            "nearest centroid on standardized wet-minus-dry acoustic feature "
            "deltas (duration excluded)"
        ),
        "split_unit": (
            "source recording and room (neither appears in both sets)"
        ),
        "held_out_source_recordings": len(held_out_sources),
        "held_out_rooms": len(held_out_rooms),
        "training_source_recordings": len(
            {row["utterance_id"] for row in training_rows}
        ),
        "test_source_recordings": len(
            {row["utterance_id"] for row in test_rows}
        ),
        "training_rooms": len({row["room_id"] for row in training_rows}),
        "test_rooms": len({row["room_id"] for row in test_rows}),
        "training_pairs": len(training_rows),
        "test_pairs": len(test_rows),
        "unused_cross_pairs": len(rows) - len(training_rows) - len(test_rows),
        "chance_accuracy": 1 / len(ROOM_TYPES),
        "accuracy": correct / len(test_rows),
        "confusion_matrix": confusion,
    }


def validate_rir_effects(
    manifest_path: Path,
    dataset_root: Path,
    rir_root: Path,
    output_csv: Path,
    summary_json: Path,
    samples_per_label: int = 20,
    rirs_per_room_type: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    rir_root = Path(rir_root)
    output_csv = Path(output_csv)
    summary_json = Path(summary_json)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if samples_per_label < 2:
        raise ValueError("samples_per_label must be at least 2.")
    if rirs_per_room_type < 2:
        raise ValueError(
            "rirs_per_room_type must be at least 2 for unseen-room evaluation."
        )

    selected_rows = _sample_training_rows(manifest_path, samples_per_label, seed)
    selected_rirs = _select_rirs(rir_root, rirs_per_room_type, seed)
    processor = AudioProcessor(target_sr=16_000)
    paired_rows: list[dict[str, Any]] = []

    for source in selected_rows:
        audio_path = dataset_root / source["audio_path"]
        dry_audio, _ = librosa.load(audio_path, sr=processor.target_sr, mono=True)
        dry_features = _extract_features(processor, dry_audio)
        for room_type, rir_path in selected_rirs:
            wet_audio = load_and_apply_rir(
                audio_path,
                rir_path,
                processor.target_sr,
            )
            wet_features = _extract_features(processor, wet_audio)
            paired_rows.append(
                {
                    "utterance_id": source["utterance_id"],
                    "label": source["label"].strip().lower(),
                    "room_type": room_type,
                    "room_id": f"{room_type}/{rir_path.parent.name}",
                    "rir_id": rir_path.stem,
                    "dry_duration_seconds": dry_features["duration_seconds"],
                    "wet_duration_seconds": wet_features["duration_seconds"],
                    **_paired_deltas(dry_features, wet_features),
                }
            )

    room_summary = _room_type_metrics(paired_rows)
    recognition = _recognize_room_type(paired_rows, seed)
    summary: dict[str, Any] = {
        "experiment": "paired dry and simulated-RIR convolved audio",
        "interpretation": (
            "This experiment measures the effect of applying known simulated "
            "RIRs and whether their room-size categories are distinguishable "
            "from paired feature changes on unseen speech and unseen rooms. "
            "It does not isolate an RIR from arbitrary recorded speech."
        ),
        "seed": seed,
        "samples_per_label": samples_per_label,
        "source_recordings": len(selected_rows),
        "rirs_per_room_type": rirs_per_room_type,
        "rir_count": len(selected_rirs),
        "paired_examples": len(paired_rows),
        "room_types": room_summary,
        "room_type_recognition": recognition,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(paired_rows)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_seed_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(summaries) < 2:
        raise ValueError("At least two seed runs are required for aggregation.")

    accuracies = [
        float(summary["room_type_recognition"]["accuracy"])
        for summary in summaries
    ]
    chance_accuracies = {
        float(summary["room_type_recognition"]["chance_accuracy"])
        for summary in summaries
    }
    if len(chance_accuracies) != 1:
        raise ValueError("Seed runs use inconsistent chance baselines.")
    if len(
        {
            (summary["samples_per_label"], summary["rirs_per_room_type"])
            for summary in summaries
        }
    ) != 1:
        raise ValueError("Seed runs use inconsistent experiment sizes.")

    return {
        "experiment": "multi-seed paired dry and simulated-RIR validation",
        "interpretation": (
            "Aggregate variability across independently sampled source "
            "recordings, room impulse responses, and source/room-disjoint "
            "recognition splits. This does not isolate an RIR from arbitrary "
            "recorded speech."
        ),
        "runs": [
            {
                "seed": summary["seed"],
                "accuracy": float(summary["room_type_recognition"]["accuracy"]),
                "test_pairs": int(summary["room_type_recognition"]["test_pairs"]),
                "confusion_matrix": summary["room_type_recognition"][
                    "confusion_matrix"
                ],
            }
            for summary in summaries
        ],
        "run_count": len(summaries),
        "samples_per_label_per_run": summaries[0]["samples_per_label"],
        "rirs_per_room_type_per_run": summaries[0]["rirs_per_room_type"],
        "chance_accuracy": chance_accuracies.pop(),
        "accuracy_mean": statistics.fmean(accuracies),
        "accuracy_population_std": statistics.pstdev(accuracies),
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "confusion_matrices": [
            {
                "seed": summary["seed"],
                **summary["room_type_recognition"]["confusion_matrix"],
            }
            for summary in summaries
        ],
    }


def validate_multiple_seeds(
    manifest_path: Path,
    dataset_root: Path,
    rir_root: Path,
    output_csv: Path,
    summary_json: Path,
    seeds: list[int],
    samples_per_label: int = 20,
    rirs_per_room_type: int = 4,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("Provide at least two distinct seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    output_csv = Path(output_csv)
    summary_json = Path(summary_json)
    summaries = []
    for seed in seeds:
        seed_csv = output_csv.with_name(
            f"{output_csv.stem}_seed{seed}{output_csv.suffix}"
        )
        seed_summary = summary_json.with_name(
            f"{summary_json.stem}_seed{seed}{summary_json.suffix}"
        )
        summary = validate_rir_effects(
            manifest_path,
            dataset_root,
            rir_root,
            seed_csv,
            seed_summary,
            samples_per_label=samples_per_label,
            rirs_per_room_type=rirs_per_room_type,
            seed=seed,
        )
        summaries.append(summary)
        print(
            f"seed {seed}: room-type accuracy "
            f"{summary['room_type_recognition']['accuracy']:.3f}"
        )

    aggregate = aggregate_seed_summaries(summaries)
    aggregate["seed_summaries"] = [
        str(
            Path(summary_json).with_name(
                f"{Path(summary_json).stem}_seed{seed}{Path(summary_json).suffix}"
            )
        )
        for seed in seeds
    ]
    aggregate["paired_feature_files"] = [
        str(
            Path(output_csv).with_name(
                f"{Path(output_csv).stem}_seed{seed}{Path(output_csv).suffix}"
            )
        )
        for seed in seeds
    ]
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dry and known-RIR audio features, then measure "
            "source- and room-disjoint simulated room-type recognition."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "asvspoof2019_la.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "asvspoof" / "LA" / "LA",
    )
    parser.add_argument(
        "--rir-root",
        type=Path,
        default=project_root / "data" / "rir" / "RIRS_NOISES" / "simulated_rirs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "data" / "rir_validation_pairs.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=project_root / "data" / "rir_validation_summary.json",
    )
    parser.add_argument("--samples-per-label", type=_positive_int, default=20)
    parser.add_argument("--rirs-per-room-type", type=_positive_int, default=4)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Run one deterministic experiment.",
    )
    seed_options.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run multiple independent experiments and aggregate their accuracy.",
    )
    args = parser.parse_args()

    if args.seeds is not None:
        summary = validate_multiple_seeds(
            args.manifest,
            args.dataset_root,
            args.rir_root,
            args.output_csv,
            args.summary_json,
            args.seeds,
            samples_per_label=args.samples_per_label,
            rirs_per_room_type=args.rirs_per_room_type,
        )
        print(f"Aggregate summary written to {args.summary_json}")
        print(
            f"Mean room-type accuracy: {summary['accuracy_mean']:.3f} "
            f"+/- {summary['accuracy_population_std']:.3f} "
            f"(range {summary['accuracy_min']:.3f}-"
            f"{summary['accuracy_max']:.3f}; "
            f"chance {summary['chance_accuracy']:.3f})"
        )
    else:
        summary = validate_rir_effects(
            args.manifest,
            args.dataset_root,
            args.rir_root,
            args.output_csv,
            args.summary_json,
            samples_per_label=args.samples_per_label,
            rirs_per_room_type=args.rirs_per_room_type,
            seed=args.seed,
        )
        print(f"Paired feature rows written to {args.output_csv}")
        print(f"Experiment summary written to {args.summary_json}")
        print(
            "Room-type recognition accuracy: "
            f"{summary['room_type_recognition']['accuracy']:.3f} "
            f"(chance: {summary['room_type_recognition']['chance_accuracy']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
