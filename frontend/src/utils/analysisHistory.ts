import type { AnalysisData } from '../types'

export const ANALYSIS_HISTORY_KEY = 'acousticspace.analysis-history.v1'
export const MAX_HISTORY_ENTRIES = 10

export interface AnalysisHistoryEntry {
  id: string
  filename: string
  analyzedAt: string
  data: AnalysisData
}

export interface LoadedHistory {
  entries: AnalysisHistoryEntry[]
  error: string | null
}

function isHistoryEntry(value: unknown): value is AnalysisHistoryEntry {
  if (typeof value !== 'object' || value === null) return false
  const entry = value as Record<string, unknown>
  if (
    typeof entry.id !== 'string' ||
    typeof entry.filename !== 'string' ||
    typeof entry.analyzedAt !== 'string' ||
    typeof entry.data !== 'object' ||
    entry.data === null
  ) {
    return false
  }
  const data = entry.data as Record<string, unknown>
  return (
    typeof data.metadata === 'object' &&
    data.metadata !== null &&
    typeof data.temporal_analysis === 'object' &&
    data.temporal_analysis !== null &&
    typeof data.waveform === 'object' &&
    data.waveform !== null &&
    typeof data.spectrogram === 'object' &&
    data.spectrogram !== null &&
    typeof data.acoustic_features === 'object' &&
    data.acoustic_features !== null
  )
}

export function loadAnalysisHistory(): LoadedHistory {
  try {
    const stored = window.localStorage.getItem(ANALYSIS_HISTORY_KEY)
    if (!stored) return { entries: [], error: null }

    const parsed: unknown = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return {
        entries: [],
        error: 'Saved analysis history has an invalid format and was not loaded.',
      }
    }

    const entries = parsed.filter(isHistoryEntry)
    return {
      entries: entries.slice(0, MAX_HISTORY_ENTRIES),
      error:
        entries.length === parsed.length
          ? null
          : 'Some saved history entries were invalid and were skipped.',
    }
  } catch {
    return {
      entries: [],
      error: 'Saved analysis history could not be read from this browser.',
    }
  }
}

export function saveAnalysisHistory(
  entries: AnalysisHistoryEntry[],
): string | null {
  try {
    window.localStorage.setItem(
      ANALYSIS_HISTORY_KEY,
      JSON.stringify(entries.slice(0, MAX_HISTORY_ENTRIES)),
    )
    return null
  } catch {
    return 'Analysis history could not be saved in this browser.'
  }
}
