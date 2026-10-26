import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_baseline_cnn import (
    SpectrogramCNN,
    calculate_metrics,
    load_split,
    train,
)


class BaselineCNNTests(unittest.TestCase):
    def test_loads_manifest_split_and_decodes_spectrogram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_csv = Path(temporary_directory) / "features.csv"
            with feature_csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["split", "label", "spectrogram_values_db"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train",
                        "label": "spoof",
                        "spectrogram_values_db": json.dumps(
                            np.zeros((64, 128)).tolist()
                        ),
                    }
                )
                writer.writerow(
                    {
                        "split": "dev",
                        "label": "bonafide",
                        "spectrogram_values_db": json.dumps(
                            np.ones((64, 128)).tolist()
                        ),
                    }
                )

            spectrograms, labels = load_split(feature_csv, "train")

        self.assertEqual(spectrograms.shape, (1, 64, 128))
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(spectrograms.dtype, np.float32)

    def test_cnn_returns_two_class_logits(self) -> None:
        model = SpectrogramCNN()
        logits = model(torch.zeros((2, 64, 128)))

        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_calculates_spoof_metrics(self) -> None:
        metrics = calculate_metrics(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 1]),
        )

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["bonafide_recall"], 0.5)
        self.assertEqual(metrics["spoof_precision"], 0.5)
        self.assertEqual(metrics["spoof_recall"], 0.5)
        self.assertEqual(metrics["spoof_f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_trains_and_saves_best_validation_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_csv = root / "features.csv"
            with feature_csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["split", "label", "spectrogram_values_db"],
                )
                writer.writeheader()
                for split in ("train", "dev", "eval"):
                    for label, value in (("bonafide", -70), ("spoof", -5)):
                        writer.writerow(
                            {
                                "split": split,
                                "label": label,
                                "spectrogram_values_db": json.dumps(
                                    np.full((64, 128), value).tolist()
                                ),
                            }
                        )

            model_path = root / "model.pt"
            result = train(
                feature_csv,
                model_path,
                epochs=1,
                batch_size=2,
                seed=7,
            )

            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(model_path.with_suffix(".metrics.json").is_file())

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["evaluation_samples"], 2)
        self.assertEqual(result["evaluation_metrics"]["samples"], 2)
        self.assertIn("state_dict", checkpoint)
        self.assertIn("evaluation_metrics", checkpoint)


if __name__ == "__main__":
    unittest.main()
