import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_ast import calculate_metrics, load_manifest


class ASTTrainingTests(unittest.TestCase):
    def test_loads_audio_manifest_rows_for_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            audio_paths = []
            rows = []
            for split in ("train", "dev", "eval"):
                for label in ("bonafide", "spoof"):
                    audio_path = dataset_root / split / f"{label}.flac"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_path.touch()
                    audio_paths.append(audio_path)
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "audio_path": f"{split}/{label}.flac",
                        }
                    )

            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["split", "label", "audio_path"],
                )
                writer.writeheader()
                writer.writerows(rows)

            samples = load_manifest(manifest_path, dataset_root, "train")

        self.assertEqual(len(samples), 2)
        self.assertEqual(
            samples,
            [(audio_paths[0], 0), (audio_paths[1], 1)],
        )

    def test_calculates_class_balanced_metrics(self) -> None:
        metrics = calculate_metrics(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 1]),
        )

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
