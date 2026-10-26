from typing import Any

import librosa
import numpy as np


class AudioProcessor:
    def __init__(self, target_sr: int = 16_000) -> None:
        self.target_sr = target_sr

    def extract_features(self, file_path: str) -> dict[str, Any]:
        """Extract a compact mel spectrogram and descriptive acoustic features."""
        audio, sample_rate = librosa.load(
            file_path,
            sr=self.target_sr,
            mono=True,
        )
        return self.extract_features_from_audio(audio, sample_rate)

    def extract_features_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        """Extract features from mono audio already loaded at the target rate."""
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("The audio file is empty or contains invalid samples.")

        duration = float(librosa.get_duration(y=audio, sr=sample_rate))
        n_fft = 1_024
        hop_length = 512

        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=64,
            fmax=sample_rate / 2,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        mel_db = self._resize_time_axis(mel_db, target_frames=128)
        mel_db = np.clip(mel_db, -80.0, 0.0)

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )[0]
        rms = librosa.feature.rms(
            y=audio,
            frame_length=n_fft,
            hop_length=hop_length,
        )[0]

        return {
            "metadata": {
                "sample_rate": sample_rate,
                "duration_seconds": round(duration, 2),
                "total_samples": int(audio.size),
            },
            "spectrogram": {
                "values_db": mel_db.round(1).tolist(),
                "frequency_bins": int(mel_db.shape[0]),
                "time_frames": int(mel_db.shape[1]),
                "min_db": float(np.min(mel_db)),
                "max_db": float(np.max(mel_db)),
            },
            "acoustic_features": {
                "mean_spectral_centroid_hz": float(np.mean(spectral_centroid)),
                "mean_spectral_rolloff_hz": float(np.mean(spectral_rolloff)),
                "mean_rms_energy": float(np.mean(rms)),
                "reverb_decay_proxy_seconds": self._estimate_decay_proxy(
                    audio,
                    sample_rate,
                ),
            },
            "interpretation": (
                "The decay value is an exploratory proxy from the recording's "
                "energy envelope; it is not an isolated room impulse response "
                "or a deepfake classification."
            ),
        }

    @staticmethod
    def _resize_time_axis(spectrogram: np.ndarray, target_frames: int) -> np.ndarray:
        source_frames = spectrogram.shape[1]
        source_positions = np.arange(source_frames)
        target_positions = np.linspace(0, source_frames - 1, target_frames)
        return np.vstack(
            [
                np.interp(target_positions, source_positions, row)
                for row in spectrogram
            ]
        )

    @staticmethod
    def _estimate_decay_proxy(audio: np.ndarray, sample_rate: int) -> float:
        power = np.square(audio.astype(np.float64))
        total_energy = float(np.sum(power))
        if total_energy <= np.finfo(np.float64).eps:
            return 0.0

        reverse_energy = np.cumsum(power[::-1])[::-1]
        decay_db = 10.0 * np.log10(
            np.maximum(reverse_energy / total_energy, np.finfo(np.float64).tiny)
        )
        below_threshold = np.flatnonzero(decay_db <= -20.0)
        if below_threshold.size == 0:
            return 0.0

        elapsed = float(below_threshold[0]) / sample_rate
        return round(elapsed * 3.0, 4)
