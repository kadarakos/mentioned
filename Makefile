.PHONY: lint format-check type-check test check-all

# Use 'uv run' to ensure we use the project's virtual environment
lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy .

test:
	uv run pytest --cov=mentioned

# A single command to run everything (used locally or in CI)
check-all: lint format-check type-check test
