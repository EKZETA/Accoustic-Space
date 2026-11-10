import unittest
from types import SimpleNamespace

import numpy as np
import torch

from app.services.ast_classifier import ASTClassifier


class ASTClassifierTests(unittest.TestCase):
    def test_is_disabled_without_a_configured_model(self) -> None:
        classifier = ASTClassifier()

        self.assertIsNone(classifier.predict(np.zeros(16_000), 16_000))

    def test_returns_labeled_model_probabilities(self) -> None:
        classifier = ASTClassifier()

        class FakeFeatureExtractor:
            def __call__(self, *_args, **_kwargs):
                return SimpleNamespace(input_values=torch.zeros((1, 2, 2)))

        class FakeModel:
            config = SimpleNamespace(id2label={0: "bonafide", 1: "spoof"})

            def __call__(self, **_kwargs):
                return SimpleNamespace(logits=torch.tensor([[-2.0, 2.0]]))

        classifier._feature_extractor = FakeFeatureExtractor()
        classifier._model = FakeModel()
        classifier._device = torch.device("cpu")

        result = classifier.predict(np.zeros(16_000), 16_000)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["predicted_label"], "spoof")
        self.assertGreater(result["confidence"], 0.9)
        self.assertEqual(set(result["probabilities"]), {"bonafide", "spoof"})


if __name__ == "__main__":
    unittest.main()
