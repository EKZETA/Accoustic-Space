import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_asvspoof_manifest import SPLITS, create_manifest


class ASVspoofManifestTests(unittest.TestCase):
    def test_creates_manifest_from_protocols_and_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory) / "dataset"
            protocol_directory = dataset_root / "ASVspoof2019_LA_cm_protocols"
            protocol_directory.mkdir(parents=True)

            expected_rows = []
            for index, (split, protocol_name) in enumerate(SPLITS.items()):
                utterance_id = f"LA_{split}_{index:04}"
                speaker_id = f"speaker_{index}"
                attack_id = "-" if split == "train" else f"A0{index}"
                label = "bonafide" if split == "train" else "spoof"
                protocol_path = protocol_directory / protocol_name
                protocol_path.write_text(
                    f"{speaker_id} {utterance_id} - {attack_id} {label}\n",
                    encoding="utf-8",
                )

                audio_path = (
                    dataset_root
                    / f"ASVspoof2019_LA_{split}"
                    / "flac"
                    / f"{utterance_id}.flac"
                )
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.touch()
                expected_rows.append((split, speaker_id, utterance_id, attack_id, label))

            output_path = Path(temporary_directory) / "manifest.csv"
            counts = create_manifest(dataset_root, output_path)

            self.assertEqual(counts, {"train": 1, "dev": 1, "eval": 1})
            with output_path.open(newline="", encoding="utf-8") as manifest:
                rows = list(csv.DictReader(manifest))

        self.assertEqual(len(rows), 3)
        for row, expected in zip(rows, expected_rows):
            split, speaker_id, utterance_id, attack_id, label = expected
            self.assertEqual(row["split"], split)
            self.assertEqual(row["speaker_id"], speaker_id)
            self.assertEqual(row["utterance_id"], utterance_id)
            self.assertEqual(row["attack_id"], "" if attack_id == "-" else attack_id)
            self.assertEqual(row["label"], label)
            self.assertEqual(
                row["audio_path"],
                f"ASVspoof2019_LA_{split}/flac/{utterance_id}.flac",
            )

    def test_reports_protocol_entries_without_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            protocol_directory = dataset_root / "ASVspoof2019_LA_cm_protocols"
            protocol_directory.mkdir()
            for index, protocol_name in enumerate(SPLITS.values()):
                (protocol_directory / protocol_name).write_text(
                    f"speaker LA_missing_{index} - - bonafide\n",
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(FileNotFoundError, "3 protocol entries"):
                create_manifest(dataset_root, dataset_root / "manifest.csv")


if __name__ == "__main__":
    unittest.main()
