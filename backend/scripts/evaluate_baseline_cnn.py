"""Evaluate a saved CNN checkpoint on a feature CSV without retraining."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch import nn

from scripts.train_baseline_cnn import (
    FEATURE_COLUMNS,
    LABEL_TO_INDEX,
    SPECTROGRAM_SHAPE,
    SpectrogramCNN,
    calculate_metrics,
)


def _iter_feature_batches(
    feature_csv: Path,
    split: str,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    with feature_csv.open(newline="", encoding="utf-8") as feature_file:
        reader = csv.DictReader(feature_file)
        missing_columns = FEATURE_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Feature CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        spectrograms: list[np.ndarray] = []
        labels: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            if row["split"] != split:
                continue
            label = row["label"].strip().lower()
            if label not in LABEL_TO_INDEX:
                raise ValueError(
                    f"Unknown label {row['label']!r} at CSV row {row_number}."
                )
            try:
                spectrogram = np.asarray(
                    json.loads(row["spectrogram_values_db"]),
                    dtype=np.float32,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid spectrogram at CSV row {row_number}."
                ) from error
            if spectrogram.shape != SPECTROGRAM_SHAPE:
                raise ValueError(
                    f"Expected spectrogram shape {SPECTROGRAM_SHAPE} at CSV "
                    f"row {row_number}, got {spectrogram.shape}."
                )
            if not np.isfinite(spectrogram).all():
                raise ValueError(
                    f"Spectrogram contains invalid values at CSV row {row_number}."
                )

            spectrograms.append(spectrogram)
            labels.append(LABEL_TO_INDEX[label])
            if len(labels) == batch_size:
                yield (
                    torch.from_numpy(np.stack(spectrograms)),
                    torch.tensor(labels, dtype=torch.int64),
                )
                spectrograms.clear()
                labels.clear()

        if labels:
            yield (
                torch.from_numpy(np.stack(spectrograms)),
                torch.tensor(labels, dtype=torch.int64),
            )


def evaluate_checkpoint(
    feature_csv: Path,
    checkpoint_path: Path,
    output_json: Path,
    split: str = "eval",
    batch_size: int = 64,
) -> dict[str, Any]:
    feature_csv = Path(feature_csv)
    checkpoint_path = Path(checkpoint_path)
    output_json = Path(output_json)
    if not feature_csv.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if split not in {"train", "dev", "eval"}:
        raise ValueError("split must be train, dev, or eval.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a model state_dict.")
    if checkpoint.get("classes") != LABEL_TO_INDEX:
        raise ValueError("Checkpoint classes do not match the baseline labels.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for spectrograms, targets in _iter_feature_batches(
        feature_csv,
        split,
        batch_size,
    ):
        targets = targets.to(device)
        with torch.no_grad():
            logits = model(spectrograms.to(device))
            loss = loss_function(logits, targets)
        total_loss += float(loss) * targets.size(0)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.cpu().numpy())

    if not labels:
        raise ValueError(f"No feature rows found for split {split!r}.")

    all_predictions = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    metrics = calculate_metrics(all_predictions, all_labels)
    if np.unique(all_labels).size != len(LABEL_TO_INDEX):
        raise ValueError(
            f"Split {split!r} must contain both bonafide and spoof labels."
        )

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "feature_csv": str(feature_csv),
        "split": split,
        "batch_size": batch_size,
        "evaluation_loss": total_loss / all_labels.size,
        "evaluation_metrics": metrics,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "classes": LABEL_TO_INDEX,
        "evaluation_note": (
            "Metrics apply only to the supplied feature split and preprocessing. "
            "Evaluation does not retrain the checkpoint."
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing CNN checkpoint on a selected feature split "
            "without retraining or loading the entire CSV into memory."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "rir_extracted_features.csv",
        help="Feature CSV generated by prepare_data.py.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root
        / "data"
        / "models"
        / "rir_baseline_cnn_balanced_1000_seed42.pt",
        help="Existing CNN checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data"
        / "models"
        / "rir_baseline_cnn_eval_metrics.json",
        help="Destination for evaluation metrics.",
    )
    parser.add_argument("--split", choices=("train", "dev", "eval"), default="eval")
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    args = parser.parse_args()

    report = evaluate_checkpoint(
        args.features,
        args.model,
        args.output,
        split=args.split,
        batch_size=args.batch_size,
    )
    print(f"Evaluation report written to {args.output}")
    print(
        f"samples: {report['evaluation_metrics']['samples']}; "
        f"accuracy: {report['evaluation_metrics']['accuracy']:.4f}; "
        f"balanced accuracy: "
        f"{report['evaluation_metrics']['balanced_accuracy']:.4f}; "
        f"spoof F1: {report['evaluation_metrics']['spoof_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
