---
title: Labeled
emoji: 🏷️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Mention Detector & Entity Labeler

This Space runs a FastAPI server inside a Docker container using `uv`.

### API Endpoints
- **Predict**: `POST /predict`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Docs**: `GET /docs` (Swagger UI)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Albert Einstein lived in the famous house.", "This book was written about him."]}' \
     https://kadarakos-mentioned.hf.space/predict
```
