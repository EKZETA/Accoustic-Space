import asyncio
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from starlette.datastructures import UploadFile

with patch.dict(os.environ):
    os.environ.pop("ACOUSTICSPACE_AST_MODEL_DIR", None)
    from app import main


class AudioAnalysisAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = np.zeros(16_000, dtype=np.float32)
        self.features = {
            "metadata": {
                "sample_rate": 16_000,
                "duration_seconds": 1.0,
                "total_samples": 16_000,
            }
        }

    def run_upload(self, filename: str = "sample.wav") -> dict[str, object]:
        upload = UploadFile(filename=filename, file=BytesIO(b"audio payload"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(main, "UPLOAD_DIR", Path(temporary_directory)),
                patch.object(
                    main.processor,
                    "load_audio",
                    return_value=(self.audio, 16_000),
                ),
                patch.object(
                    main.processor,
                    "extract_features_from_audio",
                    return_value=self.features.copy(),
                ),
            ):
                result = asyncio.run(main.process_audio_endpoint(upload))
                self.assertEqual(list(Path(temporary_directory).iterdir()), [])
        return result

    def test_extracts_features_without_optional_ast_prediction(self) -> None:
        with patch.object(main.classifier, "predict", return_value=None):
            result = self.run_upload()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filename"], "sample.wav")
        self.assertEqual(result["data"], self.features)
        self.assertNotIn("classification", result["data"])

    def test_includes_prediction_when_optional_ast_is_available(self) -> None:
        prediction = {
            "model": "Hugging Face AST (experimental)",
            "predicted_label": "spoof",
            "confidence": 0.8,
            "probabilities": {"bonafide": 0.2, "spoof": 0.8},
        }
        with patch.object(main.classifier, "predict", return_value=prediction):
            result = self.run_upload()

        self.assertEqual(result["data"]["classification"], prediction)

    def test_rejects_unsupported_audio_extension(self) -> None:
        with self.assertRaises(HTTPException) as error:
            asyncio.run(
                main.process_audio_endpoint(
                    UploadFile(filename="sample.txt", file=BytesIO(b"not audio"))
                )
            )

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("WAV, MP3, or FLAC", error.exception.detail)


if __name__ == "__main__":
    unittest.main()
