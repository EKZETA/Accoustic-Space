"""Fine-tune a Hugging Face Audio Spectrogram Transformer on ASVspoof audio."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parents[1]
LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}
SAMPLE_RATE = 16_000
REQUIRED_COLUMNS = {"split", "label", "audio_path"}


def load_manifest(
    manifest_path: Path,
    dataset_root: Path,
    split: str,
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        reader = csv.DictReader(manifest)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            if row["split"].strip().lower() != split:
                continue
            label = row["label"].strip().lower()
            if label not in LABEL_TO_INDEX:
                raise ValueError(
                    f"Unknown label {row['label']!r} at CSV row {row_number}."
                )
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = dataset_root / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found at CSV row {row_number}: {audio_path}"
                )
            samples.append((audio_path, LABEL_TO_INDEX[label]))

    if not samples:
        raise ValueError(f"No audio entries found for split {split!r}.")
    if len({label for _, label in samples}) != len(LABEL_TO_INDEX):
        raise ValueError(f"The {split} split must contain both labels.")
    return samples


class AudioManifestDataset(Dataset[tuple[np.ndarray, int]]):
    def __init__(self, samples: list[tuple[Path, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        import librosa

        audio_path, label = self.samples[index]
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Audio is empty or invalid: {audio_path}")
        return audio, label


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[int(label), int(prediction)] += 1

    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    bonafide_recall = (
        float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else 0.0
    )
    spoof_recall = (
        float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0
    )
    return {
        "samples": int(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": (bonafide_recall + spoof_recall) / 2,
        "bonafide_recall": bonafide_recall,
        "spoof_recall": spoof_recall,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for input_values, targets in loader:
            targets = targets.to(device)
            output = model(input_values=input_values.to(device), labels=targets)
            total_loss += float(output.loss) * targets.size(0)
            predictions.append(output.logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())

    all_predictions = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    return total_loss / all_labels.size, calculate_metrics(
        all_predictions,
        all_labels,
    )


def train(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    base_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive.")

    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_samples = load_manifest(manifest_path, dataset_root, "train")
    validation_samples = load_manifest(manifest_path, dataset_root, "dev")
    evaluation_samples = load_manifest(manifest_path, dataset_root, "eval")
    feature_extractor = ASTFeatureExtractor.from_pretrained(base_model)
    model = ASTForAudioClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_TO_INDEX),
        label2id=LABEL_TO_INDEX,
        id2label={index: label for label, index in LABEL_TO_INDEX.items()},
        ignore_mismatched_sizes=True,
    )

    def collate_audio(
        batch: list[tuple[np.ndarray, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audios, labels = zip(*batch, strict=True)
        inputs = feature_extractor(
            list(audios),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        return inputs.input_values, torch.tensor(labels, dtype=torch.long)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "collate_fn": collate_audio,
    }
    train_loader = DataLoader(
        AudioManifestDataset(train_samples),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        AudioManifestDataset(validation_samples),
        **loader_options,
    )
    evaluation_loader = DataLoader(
        AudioManifestDataset(evaluation_samples),
        **loader_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_result: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for input_values, targets in train_loader:
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(input_values=input_values.to(device), labels=targets)
            output.loss.backward()
            optimizer.step()

        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            device,
        )
        print(
            f"epoch {epoch}/{epochs} - "
            f"val_loss: {validation_loss:.4f} - "
            f"val_accuracy: {validation_metrics['accuracy']:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_result = {
                "best_epoch": epoch,
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "evaluation_samples": len(evaluation_samples),
                "base_model": base_model,
                "sample_rate": SAMPLE_RATE,
                "classes": LABEL_TO_INDEX,
                "seed": seed,
            }
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)
    evaluation_loss, evaluation_metrics = evaluate(
        model,
        evaluation_loader,
        device,
    )
    best_result["evaluation_loss"] = evaluation_loss
    best_result["evaluation_metrics"] = evaluation_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(best_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best AST model saved to {output_dir}")
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
            "Fine-tune AST on train audio, select by dev loss, and report "
            "held-out eval metrics."
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
        "--output",
        type=Path,
        default=project_root / "data" / "models" / "ast_asvspoof",
    )
    parser.add_argument(
        "--base-model",
        default="MIT/ast-finetuned-audioset-10-10-0.4593",
    )
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        args.manifest,
        args.dataset_root,
        args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
