from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scanner import DocumentScanner

@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    app.state.scanner = DocumentScanner()
    yield

app = FastAPI(title="Prompt injection classifier", version="1.1", lifespan=lifespan)

class Prompt(BaseModel):
    text: str = Field(min_length=1, max_length=16384)

class Passage(BaseModel):
    number: int = Field(description="One-based window number")
    start: int = Field(description="Start offset in input Unicode code points, inclusive")
    end: int = Field(description="End offset in input Unicode code points, exclusive")

class Prediction(BaseModel):
    label: Literal["attack", "safe"]
    attack_score: float
    confidence: float
    truncated: bool
    passages_checked: int
    highest_scoring_passage: Passage

@app.get("/health")
def health():
    return {"status": "ready"}

@app.post("/predict", response_model=Prediction)
def predict(prompt: Prompt):
    if not prompt.text.strip():
        raise HTTPException(422, "Text must contain a non-whitespace character")
    try:
        return app.state.scanner.scan(prompt.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
