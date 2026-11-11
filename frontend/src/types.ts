export interface AnalysisData {
  metadata: {
    sample_rate: number
    duration_seconds: number
    total_samples: number
  }
  temporal_analysis: {
    activity_segments: { start_seconds: number; end_seconds: number }[]
    pause_segments: { start_seconds: number; end_seconds: number }[]
    activity_segment_count: number
    pause_segment_count: number
    segments_truncated: boolean
    activity_ratio: number
    mean_pause_seconds: number
    interpretation: string
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
  classification?: {
    model: string
    predicted_label: string
    confidence: number
    probabilities: Record<string, number>
  }
}
