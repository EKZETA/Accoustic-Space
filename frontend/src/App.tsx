import React, { useState } from 'react';
import { AudioUpload } from './components/AudioUpload';

export const App: React.FC = () => {
  const [analysisData, setAnalysisData] = useState<any | null>(null);

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <header style={{ borderBottom: '1px solid #eee', paddingBottom: '10px', marginBottom: '20px' }}>
        <h1 style={{ color: '#111', margin: 0 }}>AcousticSpace Dashboard</h1>
        <p style={{ color: '#666', margin: '5px 0 0 0' }}>
          Deepfake Detection via Room Impulse Response (RIR) & Environmental Acoustics
        </p>
      </header>

      <main>
        <AudioUpload onAnalysisComplete={(data) => setAnalysisData(data)} />

        {analysisData && (
          <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
            <h2>Room Acoustic Extraction Results</h2>
            
            <h3>Audio Metadata</h3>
            <ul>
              <li>Duration: {analysisData.metadata.duration_seconds} seconds</li>
              <li>Sample Rate: {analysisData.metadata.sample_rate} Hz</li>
            </ul>

            <h3>Acoustic Features (RIR Indicators)</h3>
            <ul>
              <li>Mean Spectral Centroid: {analysisData.acoustic_features.mean_spectral_centroid.toFixed(2)} Hz</li>
              <li>Mean Spectral Rolloff: {analysisData.acoustic_features.mean_spectral_rolloff.toFixed(2)} Hz</li>
              <li>Energy Decay Index: {analysisData.acoustic_features.energy_decay_index.toFixed(4)}</li>
            </ul>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;