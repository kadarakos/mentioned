FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 1. Copy config files
COPY pyproject.toml uv.lock ./

# 2. Copy the source code EARLY 
# (This fixes the "Expected a Python module" error)
COPY src ./src
COPY README.md ./

# 3. Now run the sync 
# (uv will now find src/mentioned/__init__.py and be happy)
RUN uv sync --frozen

# 4. Pre-bake NLTK data
RUN uv run python -m nltk.downloader punkt punkt_tab

# 5. HF Space defaults
ENV PORT=7860
EXPOSE 7860

CMD ["uv", "run", "python", "-m", "uvicorn", "mentioned.app:app", "--host", "0.0.0.0", "--port", "7860"]
