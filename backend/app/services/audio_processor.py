import librosa
import numpy as np
from typing import Dict, Any

class AudioProcessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        Load audio and extract spectrogram alongside room acoustic indicators (RIR).
        """
        # 1. Load Audio File (Resample to 16kHz & Mono)
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))

        # 2. Extract Spectrogram (Short-Time Fourier Transform)
        stft = np.abs(librosa.stft(y))
        db_spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

        # 3. Extract Acoustic & Reverb Profile Heuristics
        # Spectral Centroid: Measures the center of mass of the frequency spectrum
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral Rolloff: Measures the high-frequency spread (acoustic reflection indicator)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Energy Decay / RMS: Initial estimation of room reflection level
        rms = librosa.feature.rms(y=y)[0]

        return {
            "metadata": {
                "sample_rate": sr,
                "duration_seconds": round(duration, 2),
                "total_samples": len(y)
            },
            "spectrogram_stats": {
                "shape": db_spectrogram.shape,
                "min_db": float(np.min(db_spectrogram)),
                "max_db": float(np.max(db_spectrogram)),
                "mean_db": float(np.mean(db_spectrogram))
            },
            "acoustic_features": {
                "mean_spectral_centroid": float(np.mean(spectral_centroid)),
                "mean_spectral_rolloff": float(np.mean(spectral_rolloff)),
                "energy_decay_index": float(np.mean(rms))
            }
        }