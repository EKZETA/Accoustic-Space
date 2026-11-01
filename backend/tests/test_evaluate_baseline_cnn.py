import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_baseline_cnn import evaluate_checkpoint
from scripts.train_baseline_cnn import LABEL_TO_INDEX, SpectrogramCNN


class BaselineCNNEvaluationTests(unittest.TestCase):
    def test_evaluates_checkpoint_in_batches_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_csv = root / "features.csv"
            checkpoint_path = root / "model.pt"
            output_path = root / "metrics.json"

            model = SpectrogramCNN()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": LABEL_TO_INDEX,
                    "best_epoch": 3,
                    "validation_loss": 0.25,
                },
                checkpoint_path,
            )
            with feature_csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["split", "label", "spectrogram_values_db"],
                )
                writer.writeheader()
                for split, label, value in (
                    ("train", "bonafide", -70),
                    ("eval", "bonafide", -70),
                    ("eval", "spoof", -5),
                    ("eval", "spoof", -4),
                ):
                    writer.writerow(
                        {
                            "split": split,
                            "label": label,
                            "spectrogram_values_db": json.dumps(
                                np.full((64, 128), value).tolist()
                            ),
                        }
                    )

            report = evaluate_checkpoint(
                feature_csv,
                checkpoint_path,
                output_path,
                split="eval",
                batch_size=2,
            )
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["evaluation_metrics"]["samples"], 3)
        self.assertEqual(report["checkpoint_best_epoch"], 3)
        self.assertEqual(report["split"], "eval")
        self.assertEqual(saved_report, report)

    def test_rejects_checkpoint_with_incompatible_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_csv = root / "features.csv"
            checkpoint_path = root / "model.pt"
            feature_csv.write_text(
                "split,label,spectrogram_values_db\n"
                "eval,bonafide,[]\n",
                encoding="utf-8",
            )
            torch.save(
                {
                    "state_dict": SpectrogramCNN().state_dict(),
                    "classes": {"real": 0, "fake": 1},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "classes"):
                evaluate_checkpoint(
                    feature_csv,
                    checkpoint_path,
                    root / "metrics.json",
                )


if __name__ == "__main__":
    unittest.main()
