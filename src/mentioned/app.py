import os
import gc
import nltk
from contextlib import asynccontextmanager
from typing import List
from nltk.tokenize import word_tokenize

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer


# Internal imports from your package
from mentioned.inference import (
    create_inference_model,
    compile_inference_model,
    ONNXMentionDetectorPipeline,
)

REPO_ID = "kadarakos/mention-detector-poc-dry-run"
ONNX_DIR = "model_v1_onnx"
MODEL_PATH = os.path.join(ONNX_DIR, "model.onnx")

# We use a global dict to store the pipeline after the heavy startup
state = {}


def ensure_nltk_resources():
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            print(f"gettin' {res} for ya...")
            nltk.download(res)


ensure_nltk_resources()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        print(f"🏗️ Compiling model from {REPO_ID}...")
        torch_model = create_inference_model(REPO_ID, "model_v1")
        compile_inference_model(torch_model, MODEL_PATH)
        # Keep tokenizer, evict Torch
        tokenizer = torch_model.tokenizer
        del torch_model
        gc.collect()
        print("✅ Compilation complete. RAM cleared.")
    else:
        print("🚀 Loading existing ONNX model...")
        tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)

    state["pipeline"] = ONNXMentionDetectorPipeline(
        MODEL_PATH,
        tokenizer,
        # TODO Don't hardcode!
        threshold=0.3,
    )
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)


class TextRequest(BaseModel):
    texts: List[str]


@app.post("/predict")
async def predict(request: TextRequest):
    docs = [word_tokenize(text) for text in request.texts]
    # docs = [text.split() for text in request.texts]
    results = state["pipeline"].predict(docs)
    print("YEAH")
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "onnx_exists": os.path.exists(MODEL_PATH)}
