import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.sample_balanced_manifest import create_balanced_manifest


class BalancedManifestTests(unittest.TestCase):
    def test_creates_reproducible_balanced_sample_for_each_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.csv"
            rows = [
                {
                    "split": split,
                    "label": label,
                    "utterance_id": f"{split}-{label}-{index}",
                }
                for split in ("train", "dev", "eval")
                for label in ("bonafide", "spoof")
                for index in range(8)
            ]
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["split", "label", "utterance_id"],
                )
                writer.writeheader()
                writer.writerows(rows)

            first_path = root / "sample-first.csv"
            second_path = root / "sample-second.csv"
            counts = create_balanced_manifest(
                manifest_path,
                first_path,
                samples_per_class=3,
                seed=7,
            )
            create_balanced_manifest(
                manifest_path,
                second_path,
                samples_per_class=3,
                seed=7,
            )

            with first_path.open(newline="", encoding="utf-8") as sample:
                sampled_rows = list(csv.DictReader(sample))
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

        self.assertEqual(len(sampled_rows), 18)
        self.assertEqual(set(counts.values()), {3})
        distribution = Counter((row["split"], row["label"]) for row in sampled_rows)
        self.assertEqual(set(distribution.values()), {3})

    def test_requires_enough_rows_for_every_split_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["split", "label"],
                )
                writer.writeheader()
                writer.writerow({"split": "train", "label": "bonafide"})

            with self.assertRaisesRegex(ValueError, "Not enough rows"):
                create_balanced_manifest(
                    manifest_path,
                    root / "sample.csv",
                    samples_per_class=1,
                )


if __name__ == "__main__":
    unittest.main()
