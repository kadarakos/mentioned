FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Stay in root to keep paths simple
WORKDIR /

# 1. Install dependencies (Cached layer)
# We need --extra train because we need Torch for the initial compilation
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra train

# 2. Pre-bake NLTK data so it doesn't download on every request
RUN uv run python -m nltk.downloader punkt punkt_tab

# 3. Copy only the source code (Excludes ONNX via .dockerignore)
COPY src ./src
COPY README.md ./

# 4. Final project install
RUN uv sync --frozen --extra train

# 5. HF Space defaults
ENV PORT=7860
EXPOSE 7860

# Run the app. The 'lifespan' in mentioned.app will handle the download/ONNX export.
CMD ["uv", "run", "python", "-m", "uvicorn", "mentioned.app:app", "--host", "0.0.0.0", "--port", "7860"]
