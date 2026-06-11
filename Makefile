.PHONY: install run test test-cov lint fmt clean pre-commit

install:
	uv sync

run:
	uv run python -m anomaly_detection.main

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src/anomaly_detection --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .venv/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
