import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AcousticSpace API",
    description="Audio feature extraction for the AcousticSpace analyst dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac"}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
processor = AudioProcessor(target_sr=16_000)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "online", "service": "AcousticSpace API", "version": "1.0.0"}


@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/extract-features")
async def process_audio_endpoint(file: UploadFile = File(...)) -> dict[str, object]:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please use WAV, MP3, or FLAC.",
        )

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            prefix="acousticspace-",
            dir=UPLOAD_DIR,
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio file exceeds the 25 MB upload limit.",
                    )
                temporary_file.write(chunk)

        features = processor.extract_features(temporary_path)
        return {
            "filename": filename,
            "status": "success",
            "data": features,
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Audio feature extraction failed for %s", filename)
        raise HTTPException(
            status_code=422,
            detail="The audio file could not be decoded or processed.",
        ) from error
    finally:
        await file.close()
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
