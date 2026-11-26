import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = Path(__file__).resolve().parent / "model"

@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True).eval()
    app.state.settings = json.loads((MODEL / "decision.json").read_text())
    yield

app = FastAPI(title="Prompt injection classifier", version="1.0", lifespan=lifespan)

class Prompt(BaseModel):
    text: str = Field(min_length=1, max_length=16384)

class Prediction(BaseModel):
    label: Literal["attack", "safe"]
    attack_score: float
    confidence: float
    truncated: bool

@app.get("/health")
def health():
    return {"status": "ready"}

@app.post("/predict", response_model=Prediction)
@torch.inference_mode()
def predict(prompt: Prompt):
    if not prompt.text.strip():
        raise HTTPException(422, "Text must contain a non-whitespace character")
    tokenizer = app.state.tokenizer
    settings = app.state.settings
    encoded = tokenizer(prompt.text, return_tensors="pt", truncation=True,
                        max_length=settings["max_tokens"], return_length=True)
    length = encoded.pop("length").item()
    truncated = length == settings["max_tokens"] and len(tokenizer.encode(
        prompt.text, verbose=False)) > length
    logits = app.state.model(**encoded).logits
    score = logits.softmax(-1)[0, 1].item()
    attack = score >= settings["threshold"]
    return Prediction(label="attack" if attack else "safe", attack_score=score,
                      confidence=score if attack else 1 - score, truncated=truncated)
