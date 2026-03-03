import time
import os
import gc
import json
import nltk
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from nltk.tokenize import word_tokenize

# Internal package imports
from mentioned.inference import (
    create_inference_model,
    compile_detector,
    compile_labeler,
    ONNXMentionDetectorPipeline,
    ONNXMentionLabelerPipeline,
    InferenceMentionLabeler
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


MENTION_CONFIDENCE = Histogram(
    "mention_detector_confidence",
    "Distribution of prediction confidence scores for detector.",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
)
ENTITY_CONFIDENCE = Histogram(
    "entity_labeler_confidence",
    "Distribution of prediction confidence scores for labeler."
)
ENTITY_LABEL_COUNTS = Counter(
    "entity_label_total",
    "Total count of predicted entity labels",
    ["label_name"]
)
INPUT_TOKENS = Histogram(
    "mention_input_tokens_count",
    "Number of tokens per input document",
    buckets=[1, 5, 10, 20, 50, 100, 250, 500]
)
MENTION_DENSITY = Histogram(
    "mention_density_ratio",
    "Ratio of mentions to total tokens in a document",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5]
)
MENTIONS_PER_DOC = Histogram(
    "mention_detector_count",
    "Number of mentions detected per document",
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Time spent in the model prediction pipeline",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

REPO_ID = os.getenv("REPO_ID", "kadarakos/entity-labeler-poc")
ENCODER_ID = os.getenv("ENCODER_ID", "distilroberta-base")
MODEL_FACTORY = os.getenv("MODEL_FACTORY", "model_v2")
DATA_FACTORY = os.getenv("DATA_FACTORY", "litbank_entities")
ENGINE_DIR = "model_v2_artifact"
MODEL_PATH = os.path.join(ENGINE_DIR, "model.onnx")

state = {}
setup_nltk()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """JIT compilation and loading for both Detector and Labeler."""

    if not os.path.exists(MODEL_PATH):
        print(f"🏗️ Engine not found. Compiling {MODEL_FACTORY} from {REPO_ID}...")
        torch_model = create_inference_model(REPO_ID, ENCODER_ID, MODEL_FACTORY, DATA_FACTORY)

        if isinstance(torch_model, InferenceMentionLabeler):
            compile_labeler(torch_model, ENGINE_DIR)
            with open(os.path.join(ENGINE_DIR, "config.json"), "w") as f:
                json.dump({"id2label": torch_model.id2label, "type": "labeler"}, f)
        else:
            compile_detector(torch_model, ENGINE_DIR)
            with open(os.path.join(ENGINE_DIR, "config.json"), "w") as f:
                json.dump({"type": "detector"}, f)

        tokenizer = torch_model.tokenizer
        del torch_model
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    with open(os.path.join(ENGINE_DIR, "config.json"), "r") as f:
        config = json.load(f)

    if config.get("type") == "labeler":
        id2label = {int(k): v for k, v in config["id2label"].items()}
        state["pipeline"] = ONNXMentionLabelerPipeline(MODEL_PATH, tokenizer, id2label)
    else:
        state["pipeline"] = ONNXMentionDetectorPipeline(MODEL_PATH, tokenizer)

    yield
    state.clear()

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.post("/predict")
async def predict(request: TextRequest):
    docs = [word_tokenize(t) for t in request.texts]
    start_time = time.perf_counter()
    results = state["pipeline"].predict(docs)
    INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

    for doc, doc_mentions in zip(docs, results):
        token_count = len(doc)
        mention_count = len(doc_mentions)
        
        # Input/Density metrics
        INPUT_TOKENS.observe(token_count)
        MENTIONS_PER_DOC.observe(mention_count)
        if token_count > 0:
            MENTION_DENSITY.observe(mention_count / token_count)

        for m in doc_mentions:
            # Basic detector confidence
            MENTION_CONFIDENCE.observe(m.get("score", 0))
            
            # Labeler specific metrics
            if "label" in m:
                ENTITY_LABEL_COUNTS.labels(label_name=m["label"]).inc()
                # Ensure we only observe label_score if it exists in the output
                if "label_score" in m:
                    ENTITY_CONFIDENCE.observe(m["label_score"])
                
    return {"results": results, "model_repo": REPO_ID}


@app.get("/")
def home():
    return {
        "message": "Mention Detector and Labeler API.",
        "docs": "/docs",
        "metrics": "/metrics",
    }
