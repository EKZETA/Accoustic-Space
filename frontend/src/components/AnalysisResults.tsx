import { Activity, BrainCircuit } from 'lucide-react'
import type { AnalysisData } from '../types'

export function AnalysisResults({ data }: { data: AnalysisData }) {
  const { temporal_analysis: temporal, metadata } = data
  const duration = Math.max(metadata.duration_seconds, 0.01)

  return (
    <section className="panel results-panel" aria-labelledby="results-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">WEEK 03 · MODEL OUTPUT</span>
          <h2 id="results-title">Analysis results</h2>
        </div>
        {data.classification && (
          <span className="result-tag">EXPERIMENTAL MODEL</span>
        )}
      </div>

      {data.classification ? (
        <div className="classification-result">
          <div className="classification-heading">
            <BrainCircuit size={17} />
            <span>{data.classification.model}</span>
          </div>
          <div className="prediction-row">
            <span>Model output</span>
            <strong>{data.classification.predicted_label}</strong>
          </div>
          <div className="confidence-label">
            <span>Maximum class probability</span>
            <strong>{(data.classification.confidence * 100).toFixed(1)}%</strong>
          </div>
          <div
            className="confidence-track"
            role="meter"
            aria-label="Maximum class probability"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(data.classification.confidence * 100)}
          >
            <span
              style={{
                width: `${Math.min(100, Math.max(0, data.classification.confidence * 100))}%`,
              }}
            />
          </div>
          <p className="result-disclaimer">
            Model probability is not calibrated certainty or forensic proof.
          </p>
        </div>
      ) : (
        <p className="model-unavailable">
          No AST model is loaded. Configure{' '}
          <code>ACOUSTICSPACE_AST_MODEL_DIR</code> to enable experimental
          classification.
        </p>
      )}

      <div className="temporal-summary">
        <div className="temporal-heading">
          <Activity size={16} />
          <strong>Activity and quiet intervals</strong>
        </div>
        <div className="temporal-stats">
          <span>
            Activity <strong>{(temporal.activity_ratio * 100).toFixed(0)}%</strong>
          </span>
          <span>
            Mean quiet interval{' '}
            <strong>{temporal.mean_pause_seconds.toFixed(2)} s</strong>
          </span>
        </div>
        <div
          className="activity-timeline"
          role="img"
          aria-label={`${temporal.activity_segment_count} energy activity intervals across ${metadata.duration_seconds} seconds`}
        >
          {temporal.activity_segments.map((segment, index) => {
            const left = (segment.start_seconds / duration) * 100
            const width =
              ((segment.end_seconds - segment.start_seconds) /
                duration) *
              100
            return (
              <span
                key={`${segment.start_seconds}-${index}`}
                className="activity-timeline-segment"
                style={{ left: `${left}%`, width: `${width}%` }}
              />
            )
          })}
        </div>
        <div className="timeline-axis">
          <span>0.0 s</span>
          <span>{metadata.duration_seconds.toFixed(2)} s</span>
        </div>
        <p className="result-disclaimer">{temporal.interpretation}</p>
        {temporal.segments_truncated && (
          <p className="result-disclaimer">
            Timeline shows the first 500 intervals; counts include all intervals.
          </p>
        )}
      </div>
    </section>
  )
}
