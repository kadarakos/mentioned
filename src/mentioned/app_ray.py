import os
import gc
import json
import torch
import nltk
from typing import List
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

from mentioned.inference import (
    create_inference_model, compile_detector, compile_labeler,
    ONNXMentionDetectorPipeline, ONNXMentionLabelerPipeline, InferenceMentionLabeler
)

# Initialize FastAPI for custom endpoints/metrics
app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
    autoscaling_config={"min_replicas": 1, "max_replicas": 5}
)
@serve.ingress(app)
class MentionService:
    def __init__(self):
        setup_nltk()
        self.engine_dir = "model_v2_artifact"
        self.model_path = os.path.join(self.engine_dir, "model.onnx")
        if not os.path.exists(self.model_path):
            self._compile_model()

        self.tokenizer = AutoTokenizer.from_pretrained(self.engine_dir)
        with open(os.path.join(self.engine_dir, "config.json"), "r") as f:
            config = json.load(f)

        if config.get("type") == "labeler":
            id2label = {int(k): v for k, v in config["id2label"].items()}
            self.pipeline = ONNXMentionLabelerPipeline(self.model_path, self.tokenizer, id2label)
        else:
            self.pipeline = ONNXMentionDetectorPipeline(self.model_path, self.tokenizer)

    def _compile_model(self):
        print("🏗️ Compiling engine...")
        torch_model = create_inference_model(
            os.getenv("REPO_ID"), os.getenv("ENCODER_ID"), 
            os.getenv("MODEL_FACTORY"), os.getenv("DATA_FACTORY")
        )
        if isinstance(torch_model, InferenceMentionLabeler):
            compile_labeler(torch_model, self.engine_dir)
            with open(os.path.join(self.engine_dir, "config.json"), "w") as f:
                json.dump({"id2label": torch_model.id2label, "type": "labeler"}, f)
        else:
            compile_detector(torch_model, self.engine_dir)
            with open(os.path.join(self.engine_dir, "config.json"), "w") as f:
                json.dump({"type": "detector"}, f)
        del torch_model
        gc.collect()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def handle_batch(self, docs_list: List[List[List[str]]]):
        # Flatten across requests.
        flattened_docs = [doc for user_docs in docs_list for doc in user_docs]
        all_results = self.pipeline.predict(flattened_docs)
        # Split back it back into per request.
        pointer = 0
        returned_results = []
        for user_docs in docs_list:
            size = len(user_docs)
            returned_results.append(all_results[pointer : pointer + size])
            pointer += size
        return returned_results

    @app.post("/predict")
    async def predict(self, request: Request):
        data = await request.json()
        texts = data.get("texts", [])
        docs = [word_tokenize(t) for t in texts]
        results = await self.handle_batch(docs)
        return {"results": results}

# Entry point for Ray Serve
mention_app = MentionService.bind()
