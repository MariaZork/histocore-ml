.PHONY: help install install-dev test lint format clean docs

help:
	@echo "HistoCore-ML Development Commands"
	@echo "================================"
	@echo "install      - Install package"
	@echo "install-dev  - Install with dev dependencies"
	@echo "test         - Run tests"
	@echo "test-cov     - Run tests with coverage"
	@echo "lint         - Run linters (ruff, mypy)"
	@echo "format       - Format code (black, ruff)"
	@echo "clean        - Clean build artifacts"
	@echo "docs         - Build documentation"
	@echo "docs-serve   - Serve docs locally"

install:
	pip install -e ".[openslide]"

install-dev:
	pip install -e ".[all]"

install-foundation:
	pip install -e ".[foundation]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=histocoreml --cov-report=html --cov-report=term

lint:
	ruff check histocoreml/ tests/ examples/
	mypy histocoreml/ --ignore-missing-imports

format:
	black histocoreml/ tests/ examples/
	ruff check --fix histocoreml/ tests/ examples/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docs:
	mkdocs build

docs-serve:
	mkdocs serve

docker-build:
	docker build -t histocoreml:latest .

docker-run:
	docker run -it --rm -v $(PWD)/data:/data histocoreml:latest

benchmark:
	python scripts/benchmark.py