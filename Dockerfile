FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app


ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./
RUN uv sync --frozen
RUN uv run python -m nltk.downloader punkt punkt_tab

ENV PORT=7860
EXPOSE 7860

# Run the app. The 'lifespan' in mentioned.app will handle the download/ONNX export.
CMD ["uv", "run", "python", "-m", "uvicorn", "mentioned.app:app", "--host", "0.0.0.0", "--port", "7860"]
