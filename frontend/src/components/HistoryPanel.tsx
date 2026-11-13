import { Clock3, Trash2 } from 'lucide-react'
import type { AnalysisHistoryEntry } from '../utils/analysisHistory'

interface HistoryPanelProps {
  entries: AnalysisHistoryEntry[]
  error: string | null
  onSelect: (entry: AnalysisHistoryEntry) => void
  onDelete: (id: string) => void
}

export function HistoryPanel({
  entries,
  error,
  onSelect,
  onDelete,
}: HistoryPanelProps) {
  return (
    <section className="panel history-panel" aria-labelledby="history-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">THIS BROWSER ONLY</span>
          <h2 id="history-title">Analysis history</h2>
        </div>
        <Clock3 size={16} aria-hidden="true" />
      </div>
      {error && <p className="history-error" role="status">{error}</p>}
      {entries.length ? (
        <ul className="history-list">
          {entries.map((entry) => (
            <li className="history-item" key={entry.id}>
              <button
                className="history-select"
                type="button"
                onClick={() => onSelect(entry)}
                title={`Load analysis for ${entry.filename}`}
              >
                <strong>{entry.filename}</strong>
                <span>
                  {new Date(entry.analyzedAt).toLocaleString()} ·{' '}
                  {entry.data.metadata.duration_seconds.toFixed(2)} s
                </span>
              </button>
              <button
                className="history-delete"
                type="button"
                onClick={() => onDelete(entry.id)}
                aria-label={`Remove ${entry.filename} from history`}
              >
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted-copy">Completed analyses will be saved here.</p>
      )}
      <p className="history-privacy">
        History stores extracted results in local browser storage, not the audio.
      </p>
    </section>
  )
}
