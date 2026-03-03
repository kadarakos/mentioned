---
title: Labeled
emoji: 🏷️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


### API Endpoints
- **Predict**: `POST /predict`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Docs**: `GET /docs` (Swagger UI)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

#Mention Detector and Entity Labeler
The project is training a `mention-detector --> entity-labeler` pipeline that share
an underlying `transformer`. It uses `lightning` to set up the training. The experiment
used the `LitBank` coreference and entity annotation using the `datasets` API from `HuggingFace`.
Logging is configured with `wandb` and models are pushed to `HuggingFaceHub`. The models 
are compiled with `onnx` for inference. The deployment is through `HuggingFaceSpace` using 
the `Dockerfile` method. The models is served in a simple `FastAPI` service. 
A separate `monitoring` space is created to host a `prometheus` server for logging. This can
be connected with `Grafana` or other scrapers. Package management with `uv` of course :-).
The CI is a simple `Github Action`. The tests are largely auto generated with Gemini and then
fixed by hand, so they are far from high coverage.

The model architecture is simple and based on the "encoder into pairwise token classifier" paradigm
from 2020: https://aclanthology.org/2020.acl-main.577/. For mention detection it uses a start
classifier that decides whether a token is a start of a span. For each start token then it 
classifies the rest of the tokens in the sentence whether they form a complete span. This way
the architecture is natively capable  of handling nested span-extraction. This is the method used
in the latest SOTA coreference system for mention-extraction: https://aclanthology.org/2025.emnlp-main.1737.pdf.
The entity-labeler also takes the start and end token pairs and for the ones that form a 
span predicts an entity label. The current implementation of the architecture uses an
MLP for pairwise token classification. But actually the `Biaffine` I tried and its better,
but it didn't make it into the codebase: https://api.wandb.ai/links/kadar-akos/5z8lkt8i.
Experiments were run on a T4 and take about 30 - 50min. The model is based on the `base`
version of `DistillRoBERTa`.

The mention-detection experiment I would consider a success with about 75% - 80% F1.
The entity-detection and labeling on the otherhand is only about 65%. Based on the paper
I guess this is expected where they get only 68%: https://aclanthology.org/N19-1220.pdf.

You can try the endpoints simply with `curl`. Here is an example for `mention-detection`:
```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Albert Einstein lived in the famous house.", "This book was written about him."]}' \
     https://kadarakos-mentioned.hf.space/predict
```

And this one for labeling:


```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Albert Einstein lived in the famous house.", "This book was written about him."]}' \
     https://kadarakos-labeled.hf.space/predict
```
