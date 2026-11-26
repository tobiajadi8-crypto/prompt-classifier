"""Regression tests with the checked-in weights; run with python -m unittest."""
import unittest

import torch
from fastapi.testclient import TestClient

from app import app


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app).__enter__()
        cls.scanner = app.state.scanner

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def scan(self, text):
        response = self.client.post("/predict", json={"text": text})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @torch.inference_mode()
    def test_short_input_preserves_original_scores_and_offsets(self):
        text = "  📋 A short résumé of the meeting.  "
        result = self.scan(text)
        tokenizer = self.scanner.tokenizer
        encoded = tokenizer(text, max_length=192, truncation=True, return_tensors="pt")
        old_score = self.scanner.model(**encoded).logits.softmax(-1)[0, 1].item()
        self.assertAlmostEqual(result["attack_score"], old_score, places=6)
        self.assertEqual(result["passages_checked"], 1)
        peak = result["highest_scoring_passage"]
        self.assertEqual(text[peak["start"]:peak["end"]], text.strip())
        self.assertFalse(result["truncated"])

    def test_window_boundaries(self):
        for length, expected in [(190, 1), (191, 2), (316, 2), (317, 3)]:
            with self.subTest(length=length):
                self.assertEqual(self.scan("note " * length)["passages_checked"], expected)

    @torch.inference_mode()
    def test_peak_matches_independent_windows_across_batches(self):
        text = ("📋 field note " * 440) + "End of report."
        result = self.scan(text)
        enc = self.scanner.tokenizer(text, max_length=192, truncation=True, stride=64,
                                     return_overflowing_tokens=True, return_offsets_mapping=True)
        scores = []
        for i in range(len(enc["input_ids"])):
            ids = {k: torch.tensor([enc[k][i]]) for k in self.scanner.tokenizer.model_input_names}
            scores.append(self.scanner.model(**ids).logits.softmax(-1)[0, 1].item())
        self.assertGreater(len(scores), 8)
        self.assertEqual(result["passages_checked"], len(scores))
        peak = result["highest_scoring_passage"]
        self.assertAlmostEqual(result["attack_score"], max(scores), places=5)
        i = peak["number"] - 1
        self.assertAlmostEqual(scores[i], max(scores), places=5)
        spans = [(a,b) for a,b in enc["offset_mapping"][i] if a < b]
        self.assertEqual((peak["start"], peak["end"]), (spans[0][0], spans[-1][1]))
        self.assertEqual(enc["offset_mapping"][-1][-2][1], len(text))

    def test_input_limits(self):
        for text in ["", "  ", "\x00", "x" * 16385]:
            self.assertEqual(self.client.post("/predict", json={"text":text}).status_code, 422)
        self.assertGreater(self.scan("a " * 8192)["passages_checked"], 1)
