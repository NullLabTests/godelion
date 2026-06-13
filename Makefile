.PHONY: install dev install-dev test lint typecheck clean docker-build docker-run run analysis

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,analysis]"
	pre-commit install

# Development
test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy godelion/ --ignore-missing-imports

lint-fix:
	ruff check --fix .
	ruff format .

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info

clean-output:
	rm -rf output_godelion/ output_dgm/ output_selfimprove/

# Docker
docker-build:
	docker build -t godelion .

docker-run:
	docker run -it --rm \
		-v $$(pwd):/godelion \
		-e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
		-e OPENAI_API_KEY=$$OPENAI_API_KEY \
		godelion

# Godelion
run:
	python run.py

run-analysis:
	python -m analysis.report --output-dir $(OUTPUT)

# Safety
check-patches:
	@echo "Checking for large patch files..."
	@find . -name "*.diff" -size +100k -exec ls -lh {} \;
