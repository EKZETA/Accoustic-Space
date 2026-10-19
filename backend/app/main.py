import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.audio_processor import AudioProcessor

app = FastAPI(
    title="AcousticSpace API",
    description="API Gateway for Deepfake Detection via Room Impulse Response (RIR)",
    version="1.0.0"
)

# CORS configuration for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
processor = AudioProcessor(target_sr=16000)

@app.get("/")
def read_root():
    return {"status": "AcousticSpace API Online", "version": "1.0.0"}

@app.post("/api/v1/extract-features")
async def process_audio_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please use WAV, MP3, or FLAC."
        )
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        features = processor.extract_features(file_path)
        return {
            "filename": file.filename,
            "status": "success",
            "data": features
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process audio: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)