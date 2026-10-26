import { useRef, useState } from 'react'
import { AlertCircle, AudioLines, FileAudio, LoaderCircle, Upload } from 'lucide-react'
import type { AnalysisData } from '../types'

interface AudioUploadProps {
  onAnalysisComplete: (filename: string, data: AnalysisData) => void
}

const MAX_FILE_SIZE = 25 * 1024 * 1024
const SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac']
const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export function AudioUpload({ onAnalysisComplete }: AudioUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [loading, setLoading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const chooseFile = (file?: File) => {
    if (!file) return
    const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()

    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setSelectedFile(null)
      setError('Pilih file WAV, MP3, atau FLAC.')
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      setSelectedFile(null)
      setError('Ukuran file maksimum adalah 25 MB.')
      return
    }

    setSelectedFile(file)
    setError(null)
  }

  const handleUpload = async () => {
    if (!selectedFile || loading) return

    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/extract-features`, {
        method: 'POST',
        body: formData,
      })
      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.detail ?? 'Audio gagal diproses.')
      }

      onAnalysisComplete(payload.filename, payload.data as AnalysisData)
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : 'Tidak dapat menghubungi AcousticSpace API.',
      )
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files[0])
  }

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">ANALYSIS INPUT</span>
          <h2 id="upload-title">Upload audio sample</h2>
        </div>
        <span className="file-limit">MAX 25 MB</span>
      </div>

      <div
        className={`drop-zone${isDragging ? ' is-dragging' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setIsDragging(false)
          }
        }}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          id="audio-file"
          type="file"
          accept=".wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />
        <div className="upload-icon">
          <Upload size={21} strokeWidth={1.8} />
        </div>
        {selectedFile ? (
          <div className="selected-file">
            <FileAudio size={18} />
            <span className="selected-file-name">{selectedFile.name}</span>
            <span className="selected-file-size">
              {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
            </span>
          </div>
        ) : (
          <>
            <p className="drop-title">Drop a recording here</p>
            <p className="drop-hint">or browse your device · WAV, MP3, FLAC</p>
          </>
        )}
        <button
          className="browse-button"
          type="button"
          onClick={() => inputRef.current?.click()}
        >
          {selectedFile ? 'Choose another file' : 'Browse files'}
        </button>
      </div>

      {error && (
        <p className="upload-error" role="alert">
          <AlertCircle size={16} />
          {error}
        </p>
      )}

      <div className="upload-footer">
        <p>
          <AudioLines size={15} />
          Spectrogram and acoustic indicators only
        </p>
        <button
          className="analyze-button"
          type="button"
          onClick={handleUpload}
          disabled={!selectedFile || loading}
        >
          {loading ? (
            <>
              <LoaderCircle className="spinner" size={17} />
              Extracting
            </>
          ) : (
            'Extract features'
          )}
        </button>
      </div>
    </section>
  )
}
