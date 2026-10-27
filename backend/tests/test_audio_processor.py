import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from app.services.audio_processor import AudioProcessor


class AudioProcessorTests(unittest.TestCase):
    def test_extracts_fixed_size_spectrogram_and_acoustic_features(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "tone.wav"
            wavfile.write(audio_path, sample_rate, audio)
            result = AudioProcessor().extract_features(str(audio_path))

        spectrogram = result["spectrogram"]
        self.assertEqual(result["metadata"]["sample_rate"], sample_rate)
        self.assertEqual(result["metadata"]["duration_seconds"], 1.0)
        waveform = result["waveform"]
        self.assertEqual(waveform["time_bins"], 1_200)
        self.assertEqual(len(waveform["amplitude_ranges"]), 1_200)
        self.assertTrue(
            all(
                len(amplitude_range) == 2
                and amplitude_range[0] <= amplitude_range[1]
                for amplitude_range in waveform["amplitude_ranges"]
            )
        )
        self.assertTrue(
            np.isclose(min(item[0] for item in waveform["amplitude_ranges"]), -0.2)
        )
        self.assertTrue(
            np.isclose(max(item[1] for item in waveform["amplitude_ranges"]), 0.2)
        )
        self.assertEqual(spectrogram["frequency_bins"], 64)
        self.assertEqual(spectrogram["time_frames"], 128)
        self.assertEqual(len(spectrogram["values_db"]), 64)
        self.assertTrue(
            all(len(row) == 128 for row in spectrogram["values_db"])
        )
        self.assertTrue(
            np.isfinite(result["acoustic_features"]["mean_spectral_centroid_hz"])
        )
        self.assertIn("not an isolated room impulse response", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
