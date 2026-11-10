from pathlib import Path
from typing import Any

import numpy as np
import torch

SAMPLE_RATE = 16_000


class ASTClassifier:
    def __init__(self, model_dir: str | None = None) -> None:
        self._model: Any = None
        self._feature_extractor: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_dir is None:
            return

        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"AST model directory not found: {model_path}")

        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        self._model = ASTForAudioClassification.from_pretrained(model_path).to(
            self._device
        )
        self._model.eval()

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any] | None:
        if self._model is None or self._feature_extractor is None:
            return None
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"AST inference requires {SAMPLE_RATE} Hz audio.")

        chunk_size = SAMPLE_RATE * 10
        logits_by_chunk = []
        with torch.no_grad():
            for offset in range(0, audio.size, chunk_size):
                inputs = self._feature_extractor(
                    audio[offset : offset + chunk_size],
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=1024,
                    truncation=True,
                )
                logits_by_chunk.append(
                    self._model(
                        input_values=inputs.input_values.to(self._device)
                    ).logits[0]
                )
            mean_logits = torch.stack(logits_by_chunk).mean(dim=0)
            probabilities = torch.softmax(mean_logits, dim=-1).cpu().tolist()

        labels = self._model.config.id2label
        label_probabilities = {
            str(labels.get(index, f"class_{index}")): float(probability)
            for index, probability in enumerate(probabilities)
        }
        predicted_index = int(np.argmax(probabilities))
        return {
            "model": "Hugging Face AST (experimental)",
            "predicted_label": str(
                labels.get(predicted_index, f"class_{predicted_index}")
            ),
            "confidence": float(probabilities[predicted_index]),
            "probabilities": label_probabilities,
        }
