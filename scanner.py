"""Score overlapping WordPiece windows with the bundled document classifier."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = Path(__file__).resolve().parent / "model"
OVERLAP = 64
BATCH_SIZE = 8


class DocumentScanner:
    def __init__(self, model_dir=MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).eval()
        settings = json.loads((model_dir / "decision.json").read_text())
        self.limit = settings["max_tokens"]
        self.threshold = settings["threshold"]

    @torch.inference_mode()
    def scan(self, text):
        windows = self.tokenizer(
            text, max_length=self.limit, truncation=True, stride=OVERLAP,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding=True, return_tensors="pt",
        )
        offsets = windows.pop("offset_mapping")
        windows.pop("overflow_to_sample_mapping")
        count = len(offsets)
        scores = []
        for start in range(0, count, BATCH_SIZE):
            batch = {key: value[start:start + BATCH_SIZE] for key, value in windows.items()}
            logits = self.model(**batch).logits
            scores.extend(logits.softmax(dim=-1)[:, 1].tolist())

        peak = max(range(count), key=scores.__getitem__)
        spans = offsets[peak]
        content = spans[spans[:, 1] > spans[:, 0]]
        if len(content) == 0:
            raise ValueError("Text must contain at least one token.")
        begin, end = int(content[0, 0]), int(content[-1, 1])
        score = scores[peak]
        attack = score >= self.threshold
        return {
            "label": "attack" if attack else "safe",
            "attack_score": score,
            "confidence": score if attack else 1 - score,
            "truncated": False,
            "passages_checked": count,
            "highest_scoring_passage": {"number": peak + 1, "start": begin, "end": end},
        }
