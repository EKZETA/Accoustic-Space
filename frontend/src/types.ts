export interface AnalysisData {
  metadata: {
    sample_rate: number
    duration_seconds: number
    total_samples: number
  }
  waveform: {
    amplitude_ranges: [number, number][]
    time_bins: number
  }
  spectrogram: {
    values_db: number[][]
    frequency_bins: number
    time_frames: number
    min_db: number
    max_db: number
  }
  acoustic_features: {
    mean_spectral_centroid_hz: number
    mean_spectral_rolloff_hz: number
    mean_rms_energy: number
    reverb_decay_proxy_seconds: number
  }
  interpretation: string
}
