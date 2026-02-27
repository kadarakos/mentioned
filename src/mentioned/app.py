import os
import gc
import nltk
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
from nltk.tokenize import word_tokenize

# Internal package imports
from mentioned.inference import (
    create_inference_model,
    compile_inference_model,
    ONNXMentionDetectorPipeline,
)


def setup_nltk():
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res)



class TextRequest(BaseModel):
    texts: List[str]


MODEL_CONFIDENCE = Histogram(
    "mention_detector_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
)
MENTIONS_PER_DOC = Histogram(
    "mention_detector_density",
    "Number of mentions detected per document",
    buckets=[0, 1, 2, 5, 10, 20, 50]
)
REPO_ID = os.getenv("REPO_ID", "kadarakos/mention-detector-poc-dry-run")
ENGINE_DIR = "engine"
MODEL_PATH = os.path.join(ENGINE_DIR, "model.onnx")

state = {}
setup_nltk()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the JIT compilation and RAM cleanup."""
    if not os.path.exists(MODEL_PATH):
        print(f"🏗️ Engine not found. Compiling from {REPO_ID}...")
        # create_inference_model respects HF_TOKEN env var automatically
        torch_model = create_inference_model(REPO_ID, "model_v1")
        compile_inference_model(torch_model, ENGINE_DIR)
        tokenizer = torch_model.tokenizer
        del torch_model
        gc.collect()
        print("✅ Compilation complete.")
    else:
        print("🚀 Loading existing ONNX engine...")
        tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)

    state["pipeline"] = ONNXMentionDetectorPipeline(MODEL_PATH, tokenizer)
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.post("/predict")
async def predict(request: TextRequest):
    pipeline = state["pipeline"]
    docs = [word_tokenize(t) for t in request.texts]
    batch_results = pipeline.predict(docs)
    for doc_mentions in batch_results:
        MENTIONS_PER_DOC.observe(len(doc_mentions))
        for m in doc_mentions:
            MODEL_CONFIDENCE.observe(m["score"])

    return {"results": batch_results}


@app.get("/")
def home():
    return {
        "message": "Mention Detector API",
        "docs": "/docs",
        "metrics": "/metrics",
    }
