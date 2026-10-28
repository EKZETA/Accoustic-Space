import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.validate_rir_effects import validate_rir_effects


class RIRValidationTests(unittest.TestCase):
    def test_runs_source_disjoint_paired_rir_validation(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            rir_root = root / "rirs"
            manifest_path = root / "manifest.csv"
            output_csv = root / "paired.csv"
            summary_json = root / "summary.json"
            manifest_rows = []

            for label, frequency in (("bonafide", 330), ("spoof", 880)):
                for index in range(2):
                    utterance_id = f"{label}-{index}"
                    audio_path = dataset_root / f"{utterance_id}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio = (
                        0.2 * np.sin(2 * np.pi * frequency * time)
                    ).astype(np.float32)
                    wavfile.write(audio_path, sample_rate, audio)
                    manifest_rows.append(
                        {
                            "split": "train",
                            "label": label,
                            "utterance_id": utterance_id,
                            "audio_path": audio_path.name,
                        }
                    )

            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=[
                        "split",
                        "label",
                        "utterance_id",
                        "audio_path",
                    ],
                )
                writer.writeheader()
                writer.writerows(manifest_rows)

            for room_index, room_type in enumerate(
                ("smallroom", "mediumroom", "largeroom"),
                start=1,
            ):
                for room_number in range(1, 3):
                    rir_directory = (
                        rir_root / room_type / f"Room{room_number:03}"
                    )
                    rir_directory.mkdir(parents=True)
                    rir = np.zeros(sample_rate // 20, dtype=np.float32)
                    rir[0] = 1.0
                    rir[min(room_index * room_number * 80, rir.size - 1)] = 0.5
                    wavfile.write(
                        rir_directory / f"Room{room_number:03}-00001.wav",
                        sample_rate,
                        rir,
                    )

            summary = validate_rir_effects(
                manifest_path,
                dataset_root,
                rir_root,
                output_csv,
                summary_json,
                samples_per_label=2,
                rirs_per_room_type=2,
                seed=7,
            )

            with output_csv.open(newline="", encoding="utf-8") as output:
                paired_rows = list(csv.DictReader(output))
            saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(len(paired_rows), 24)
        self.assertEqual(summary["source_recordings"], 4)
        self.assertEqual(summary["rir_count"], 6)
        self.assertEqual(summary["paired_examples"], 24)
        self.assertEqual(
            summary["room_type_recognition"]["training_pairs"]
            + summary["room_type_recognition"]["test_pairs"],
            12,
        )
        self.assertEqual(
            summary["room_type_recognition"]["split_unit"],
            "source recording and room (neither appears in both sets)",
        )
        self.assertIn(
            "duration excluded",
            summary["room_type_recognition"]["method"],
        )
        self.assertIn("does not isolate an RIR", saved_summary["interpretation"])
        self.assertTrue(
            all(
                room["pairs"] == 8
                for room in summary["room_types"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
