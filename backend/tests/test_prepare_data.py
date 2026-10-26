import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.prepare_data import generate_rir_features_from_manifest


class RIRFeaturePipelineTests(unittest.TestCase):
    def test_applies_rir_and_writes_librosa_features_to_csv(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        rir = np.zeros(sample_rate // 20, dtype=np.float32)
        rir[0] = 1.0
        rir[-1] = 0.5

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_path = root / "audio.wav"
            rir_path = root / "rir.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "features.csv"
            wavfile.write(audio_path, sample_rate, audio)
            wavfile.write(rir_path, sample_rate, rir)

            with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=[
                        "utterance_id",
                        "split",
                        "label",
                        "audio_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utterance_id": "valid-1",
                        "split": "train",
                        "label": "bonafide",
                        "audio_path": "audio.wav",
                    }
                )

            counts = generate_rir_features_from_manifest(
                manifest_path,
                root,
                rir_path,
                output_path,
            )

            with output_path.open(newline="", encoding="utf-8") as output:
                rows = list(csv.DictReader(output))

        self.assertEqual(counts, {"processed": 1, "failed": 0})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["utterance_id"], "valid-1")
        self.assertEqual(row["label"], "bonafide")
        self.assertGreater(float(row["duration_seconds"]), 0.1)
        self.assertTrue(float(row["mean_spectral_centroid_hz"]) > 0)
        self.assertTrue(float(row["mean_spectral_rolloff_hz"]) > 0)
        self.assertEqual(int(row["spectrogram_frequency_bins"]), 64)
        self.assertEqual(int(row["spectrogram_time_frames"]), 128)
        self.assertEqual(len(json.loads(row["spectrogram_values_db"])), 64)


if __name__ == "__main__":
    unittest.main()
