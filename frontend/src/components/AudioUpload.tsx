import React, { useState } from 'react';
import { Upload, FileAudio, AlertCircle } from 'lucide-react';

interface AudioUploadProps {
  onAnalysisComplete: (data: any) => void;
}

export const AudioUpload: React.FC<AudioUploadProps> = ({ onAnalysisComplete }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/v1/extract-features', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process the audio file.');
      }

      const result = await response.json();
      onAnalysisComplete(result.data);
    } catch (err: any) {
      setError(err.message || 'A system error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ border: '2px dashed #ccc', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
      <Upload size={48} style={{ color: '#0070f3', marginBottom: '10px' }} />
      <h3>Upload Forensic Audio File</h3>
      <p style={{ fontSize: '12px', color: '#666' }}>Supported formats: WAV, MP3, FLAC</p>
      
      <input 
        type="file" 
        accept="audio/*" 
        onChange={handleFileChange} 
        style={{ margin: '15px 0', display: 'block', marginLeft: 'auto', marginRight: 'auto' }} 
      />

      {selectedFile && (
        <div style={{ marginBottom: '15px' }}>
          <FileAudio size={16} style={{ display: 'inline', marginRight: '5px' }} /> 
          <span>{selectedFile.name}</span>
        </div>
      )}

      <button 
        onClick={handleUpload} 
        disabled={!selectedFile || loading}
        style={{
          backgroundColor: selectedFile && !loading ? '#0070f3' : '#ccc',
          color: '#fff',
          border: 'none',
          padding: '10px 20px',
          borderRadius: '4px',
          cursor: selectedFile && !loading ? 'pointer' : 'not-allowed'
        }}
      >
        {loading ? 'Extracting Acoustic Features...' : 'Process Audio'}
      </button>

      {error && (
        <p style={{ color: 'red', marginTop: '10px' }}>
          <AlertCircle size={16} style={{ display: 'inline', marginRight: '5px' }} /> 
          {error}
        </p>
      )}
    </div>
  );
};