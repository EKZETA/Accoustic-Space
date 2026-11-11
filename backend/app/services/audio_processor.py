from typing import Any

import librosa
import numpy as np

WAVEFORM_BINS = 1_200


class AudioProcessor:
    def __init__(self, target_sr: int = 16_000) -> None:
        self.target_sr = target_sr

    def load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """Decode audio once for feature extraction and optional model inference."""
        return librosa.load(
            file_path,
            sr=self.target_sr,
            mono=True,
        )

    def extract_features(self, file_path: str) -> dict[str, Any]:
        """Extract a compact mel spectrogram and descriptive acoustic features."""
        audio, sample_rate = self.load_audio(file_path)
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
            "temporal_analysis": self._extract_temporal_analysis(
                audio,
                sample_rate,
            ),
            "waveform": {
                "amplitude_ranges": self._create_waveform(audio),
                "time_bins": min(audio.size, WAVEFORM_BINS),
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

    @classmethod
    def _extract_temporal_analysis(
        cls,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        frame_length = max(1, round(sample_rate * 0.025))
        hop_length = max(1, round(sample_rate * 0.01))
        frame_energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        threshold = max(
            float(np.percentile(frame_energy, 20)) * 2.5,
            float(np.max(frame_energy)) * 0.03,
            1e-4,
        )
        active = frame_energy >= threshold
        frame_count = max(1, round(0.08 * sample_rate / hop_length))

        for start, end in cls._boolean_runs(~active):
            if end - start < frame_count:
                active[start:end] = True
        activity_runs = [
            (start, end)
            for start, end in cls._boolean_runs(active)
            if end - start >= frame_count
        ]

        duration = audio.size / sample_rate
        activities = [
            {
                "start_seconds": round(
                    max(
                        0.0,
                        start * hop_length / sample_rate
                        - frame_length / (2 * sample_rate),
                    ),
                    2,
                ),
                "end_seconds": round(
                    min(
                        duration,
                        end * hop_length / sample_rate
                        - frame_length / (2 * sample_rate),
                    ),
                    2,
                ),
            }
            for start, end in activity_runs
        ]
        pauses: list[dict[str, float]] = []
        cursor = 0.0
        for activity in activities:
            activity_start = activity["start_seconds"]
            if activity_start - cursor >= 0.15:
                pauses.append(
                    {
                        "start_seconds": round(cursor, 2),
                        "end_seconds": activity_start,
                    }
                )
            cursor = activity["end_seconds"]
        if duration - cursor >= 0.15:
            pauses.append(
                {
                    "start_seconds": round(cursor, 2),
                    "end_seconds": round(duration, 2),
                }
            )

        activity_duration = sum(
            item["end_seconds"] - item["start_seconds"] for item in activities
        )
        pause_durations = [
            item["end_seconds"] - item["start_seconds"] for item in pauses
        ]
        max_segments = 500
        return {
            "activity_segments": activities[:max_segments],
            "pause_segments": pauses[:max_segments],
            "activity_segment_count": len(activities),
            "pause_segment_count": len(pauses),
            "segments_truncated": (
                len(activities) > max_segments or len(pauses) > max_segments
            ),
            "activity_ratio": round(activity_duration / duration, 4),
            "mean_pause_seconds": round(
                float(np.mean(pause_durations)) if pause_durations else 0.0,
                3,
            ),
            "interpretation": (
                "Energy-thresholded activity and pause timing only; this does not "
                "detect breathing, syllables, or deepfake speech."
            ),
        }

    @staticmethod
    def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
        padded = np.concatenate(([False], values, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), ends.tolist(), strict=True))

    @staticmethod
    def _create_waveform(audio: np.ndarray) -> list[list[float]]:
        return [
            [float(np.min(chunk)), float(np.max(chunk))]
            for chunk in np.array_split(audio, min(audio.size, WAVEFORM_BINS))
        ]

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
