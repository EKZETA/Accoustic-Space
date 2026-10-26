import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AudioWaveform,
  Check,
  Database,
  Headphones,
  LayoutDashboard,
  LockKeyhole,
  Radio,
  ScanLine,
  Waves,
} from 'lucide-react'
import { AudioUpload } from './components/AudioUpload'
import type { AnalysisData } from './types'
import './App.css'

interface AnalysisResult {
  filename: string
  data: AnalysisData
}

function Spectrogram({ values }: { values: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context || values.length === 0) return

    const height = values.length
    const width = values[0]?.length ?? 0
    canvas.width = width
    canvas.height = height

    values.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        const intensity = Math.max(0, Math.min(1, (value + 80) / 80))
        const hue = 188 - intensity * 165
        const lightness = 12 + intensity * 47
        context.fillStyle = `hsl(${hue} 78% ${lightness}%)`
        context.fillRect(columnIndex, height - rowIndex - 1, 1, 1)
      })
    })
  }, [values])

  return (
    <canvas
      ref={canvasRef}
      className="spectrogram-canvas"
      role="img"
      aria-label="Mel spectrogram, low frequencies at the bottom and high frequencies at the top"
    />
  )
}

export default function App() {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const features = analysis?.data.acoustic_features
  const metadata = analysis?.data.metadata

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="AcousticSpace home">
          <span className="brand-mark">
            <AudioWaveform size={21} />
          </span>
          <span>
            <strong>acoustic<span>space</span></strong>
            <small> AUDIO FORENSICS</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-link active" href="#overview" aria-current="page">
            <LayoutDashboard size={17} />
            Overview
          </a>
          <a className="nav-link" href="#dataset">
            <Database size={17} />
            Dataset
          </a>
          <a className="nav-link" href="#pipeline">
            <Activity size={17} />
            Feature pipeline
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-wave">
            <Waves size={20} />
            <span>Room acoustics<br />tell another story.</span>
          </div>
          <div className="sidebar-version">
            <span className="online-dot" />
            WEEK 01 · FEATURE EXTRACTION
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-slash">/</span>
            <strong>Audio analysis</strong>
          </div>
          <div className="topbar-status">
            <span className="online-dot" />
            Local analysis workspace
          </div>
        </header>

        <main id="overview" className="dashboard">
          <section className="page-intro">
            <div>
              <div className="eyebrow">ACOUSTICSPACE / ANALYST CONSOLE</div>
              <h1>Audio analysis</h1>
              <p>Inspect spatial and spectral clues in a recording.</p>
            </div>
            <div className="intro-badge">
              <LockKeyhole size={15} />
              Files processed locally
            </div>
          </section>

          <div className="dashboard-grid">
            <div className="primary-column">
              <AudioUpload
                onAnalysisComplete={(filename, data) =>
                  setAnalysis({ filename, data })
                }
              />

              <section className="panel metrics-panel" aria-labelledby="metrics-title">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">SIGNAL SUMMARY</span>
                    <h2 id="metrics-title">
                      {analysis ? 'Extracted indicators' : 'Ready for a sample'}
                    </h2>
                  </div>
                  {analysis && (
                    <span className="result-tag">
                      <Check size={13} />
                      FEATURES EXTRACTED
                    </span>
                  )}
                </div>
                <div className="metric-grid">
                  <Metric
                    icon={<Headphones size={17} />}
                    label="Duration"
                    value={metadata ? `${metadata.duration_seconds.toFixed(2)} s` : '—'}
                    detail="audio length"
                  />
                  <Metric
                    icon={<Radio size={17} />}
                    label="Sample rate"
                    value={metadata ? `${(metadata.sample_rate / 1000).toFixed(1)} kHz` : '—'}
                    detail="mono · resampled"
                  />
                  <Metric
                    icon={<ScanLine size={17} />}
                    label="Spectral centroid"
                    value={
                      features
                        ? `${features.mean_spectral_centroid_hz.toFixed(0)} Hz`
                        : '—'
                    }
                    detail="mean frequency balance"
                  />
                  <Metric
                    icon={<Waves size={17} />}
                    label="Decay proxy"
                    value={
                      features
                        ? `${features.reverb_decay_proxy_seconds.toFixed(2)} s`
                        : '—'
                    }
                    detail="exploratory · not an RIR"
                  />
                </div>
              </section>

              <section className="panel spectrogram-panel" aria-labelledby="spectrogram-title">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">TIME × FREQUENCY</span>
                    <h2 id="spectrogram-title">Mel spectrogram</h2>
                  </div>
                  {analysis && (
                    <span className="spectrogram-size">
                      {analysis.data.spectrogram.frequency_bins} ×{' '}
                      {analysis.data.spectrogram.time_frames} bins
                    </span>
                  )}
                </div>
                {analysis ? (
                  <div className="spectrogram-wrap">
                    <div className="frequency-axis">
                      <span>8 kHz</span>
                      <span>4 kHz</span>
                      <span>0 Hz</span>
                    </div>
                    <Spectrogram values={analysis.data.spectrogram.values_db} />
                    <div className="time-axis">
                      <span>0.0 s</span>
                      <span>{(analysis.data.metadata.duration_seconds / 2).toFixed(1)} s</span>
                      <span>{analysis.data.metadata.duration_seconds.toFixed(1)} s</span>
                    </div>
                  </div>
                ) : (
                  <div className="spectrogram-empty">
                    <div className="empty-graphic">
                      <span /><span /><span /><span /><span /><span /><span />
                    </div>
                    <p>A frequency map will appear after feature extraction.</p>
                  </div>
                )}
                {analysis && (
                  <div className="spectrogram-legend">
                    <span>LOW ENERGY</span>
                    <i />
                    <span>HIGH ENERGY</span>
                  </div>
                )}
              </section>
            </div>

            <aside className="secondary-column">
              <section className="panel file-panel" aria-labelledby="file-title">
                <span className="eyebrow">CURRENT SAMPLE</span>
                <h2 id="file-title">{analysis ? 'Analysis file' : 'No file analyzed'}</h2>
                {analysis ? (
                  <div className="sample-file">
                    <span className="sample-file-icon"><AudioWaveform size={19} /></span>
                    <span className="sample-file-name" title={analysis.filename}>
                      {analysis.filename}
                    </span>
                    <Check className="sample-check" size={16} />
                  </div>
                ) : (
                  <p className="muted-copy">
                    Your selected audio and its extracted features will be summarized here.
                  </p>
                )}
                <div className="privacy-note">
                  <LockKeyhole size={15} />
                  Temporary upload is deleted after processing.
                </div>
              </section>

              <section id="dataset" className="panel dataset-panel">
                <span className="eyebrow">DATA PREPARATION</span>
                <h2>ASVspoof 2019 · LA</h2>
                <p className="muted-copy">
                  Protocol-based manifest for labeled bona fide and spoof recordings.
                </p>
                <div className="split-list">
                  <SplitRow name="Train" detail="protocol split" />
                  <SplitRow name="Development" detail="protocol split" />
                  <SplitRow name="Evaluation" detail="protocol split" />
                </div>
                <div className="dataset-note">
                  Generate a local CSV manifest with the backend preparation script.
                </div>
              </section>

              <section id="pipeline" className="panel pipeline-panel">
                <span className="eyebrow">WEEK 01 PIPELINE</span>
                <h2>Extraction status</h2>
                <div className="pipeline-step">
                  <span className="step-number">01</span>
                  <span className="step-copy"><strong>Audio decode</strong><small>Mono · 16 kHz</small></span>
                  <Check size={15} />
                </div>
                <div className="pipeline-step">
                  <span className="step-number">02</span>
                  <span className="step-copy"><strong>Mel spectrogram</strong><small>64 × 128 summary</small></span>
                  <Check size={15} />
                </div>
                <div className="pipeline-step">
                  <span className="step-number">03</span>
                  <span className="step-copy"><strong>Acoustic indicators</strong><small>Descriptive features</small></span>
                  <Check size={15} />
                </div>
                <div className="pipeline-disclaimer">
                  No classifier or deepfake verdict is active in Week 1.
                </div>
              </section>
            </aside>
          </div>
          <footer className="dashboard-footer">
            <span>ACOUSTICSPACE · RESEARCH PROTOTYPE</span>
            <span>Acoustic features are exploratory, not forensic proof.</span>
          </footer>
        </main>
      </div>
    </div>
  )
}

function Metric({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode
  label: string
  value: string
  detail: string
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">{icon}{label}</div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  )
}

function SplitRow({ name, detail }: { name: string; detail: string }) {
  return (
    <div className="split-row">
      <span className="split-dot" />
      <span>{name}</span>
      <small>{detail}</small>
    </div>
  )
}
