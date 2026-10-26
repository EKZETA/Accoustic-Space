"""Train a small CNN baseline on RIR-augmented mel spectrograms."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = {"split", "label", "spectrogram_values_db"}
LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}
SPECTROGRAM_SHAPE = (64, 128)


class SpectrogramCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(LABEL_TO_INDEX)),
        )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        if spectrograms.ndim == 3:
            spectrograms = spectrograms.unsqueeze(1)
        return self.classifier(self.features(spectrograms))


def load_split(
    feature_csv: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    spectrograms: list[np.ndarray] = []
    labels: list[int] = []

    with feature_csv.open(newline="", encoding="utf-8") as feature_file:
        reader = csv.DictReader(feature_file)
        missing_columns = FEATURE_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Feature CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

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

    if not spectrograms:
        raise ValueError(f"No feature rows found for split {split!r}.")
    return np.stack(spectrograms), np.asarray(labels, dtype=np.int64)


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    true_positive = int(np.sum((predictions == 1) & (labels == 1)))
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    false_negative = int(np.sum((predictions == 0) & (labels == 1)))
    true_negative = int(np.sum((predictions == 0) & (labels == 0)))
    accuracy = float(np.mean(predictions == labels))
    bonafide_recall = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    bonafide_precision = (
        true_negative / (true_negative + false_negative)
        if true_negative + false_negative
        else 0.0
    )
    bonafide_f1 = (
        2 * bonafide_precision * bonafide_recall
        / (bonafide_precision + bonafide_recall)
        if bonafide_precision + bonafide_recall
        else 0.0
    )
    return {
        "samples": int(labels.size),
        "accuracy": accuracy,
        "balanced_accuracy": (bonafide_recall + recall) / 2,
        "bonafide_precision": bonafide_precision,
        "bonafide_recall": bonafide_recall,
        "bonafide_f1": bonafide_f1,
        "spoof_precision": precision,
        "spoof_recall": recall,
        "spoof_f1": f1,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def evaluate(
    model: SpectrogramCNN,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for spectrograms, targets in loader:
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            logits = model(spectrograms)
            total_loss += float(loss_function(logits, targets)) * targets.size(0)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())

    all_predictions = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    return total_loss / all_labels.size, calculate_metrics(all_predictions, all_labels)


def train(
    feature_csv: Path,
    model_path: Path,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_spectrograms, train_labels = load_split(feature_csv, "train")
    validation_spectrograms, validation_labels = load_split(feature_csv, "dev")
    evaluation_spectrograms, evaluation_labels = load_split(feature_csv, "eval")
    if np.unique(train_labels).size != len(LABEL_TO_INDEX):
        raise ValueError("The train split must contain both bonafide and spoof labels.")
    if np.unique(validation_labels).size != len(LABEL_TO_INDEX):
        raise ValueError("The dev split must contain both bonafide and spoof labels.")
    if np.unique(evaluation_labels).size != len(LABEL_TO_INDEX):
        raise ValueError("The eval split must contain both bonafide and spoof labels.")

    train_dataset = TensorDataset(
        torch.from_numpy(train_spectrograms),
        torch.from_numpy(train_labels),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_spectrograms),
        torch.from_numpy(validation_labels),
    )
    evaluation_dataset = TensorDataset(
        torch.from_numpy(evaluation_spectrograms),
        torch.from_numpy(evaluation_labels),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_result: dict[str, Any] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for spectrograms, targets in train_loader:
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(spectrograms), targets)
            loss.backward()
            optimizer.step()

        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )
        print(
            f"epoch {epoch}/{epochs} - "
            f"val_loss: {validation_loss:.4f} - "
            f"val_accuracy: {validation_metrics['accuracy']:.4f} - "
            f"spoof_f1: {validation_metrics['spoof_f1']:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_result = {
                "best_epoch": epoch,
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "train_samples": int(train_labels.size),
                "validation_samples": int(validation_labels.size),
                "evaluation_samples": int(evaluation_labels.size),
                "classes": LABEL_TO_INDEX,
                "model_config": {
                    "input_shape": list(SPECTROGRAM_SHAPE),
                    "architecture": "SpectrogramCNN",
                },
                "seed": seed,
            }
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    model.load_state_dict(best_state_dict)
    evaluation_loss, evaluation_metrics = evaluate(
        model,
        evaluation_loader,
        loss_function,
        device,
    )
    best_result["evaluation_loss"] = evaluation_loss
    best_result["evaluation_metrics"] = evaluation_metrics
    torch.save(
        {
            **best_result,
            "state_dict": best_state_dict,
        },
        model_path,
    )
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(best_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best model saved to {model_path}")
    print(f"Validation and held-out eval metrics saved to {metrics_path}")
    return best_result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN baseline on train features, select by dev loss, "
            "and report held-out eval metrics."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "rir_extracted_features.csv",
        help="CSV generated by prepare_data.py.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "data" / "models" / "rir_baseline_cnn.pt",
        help="Output path for the best validation checkpoint.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=15)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        args.features,
        args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
