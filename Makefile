.DEFAULT_GOAL := help

.PHONY: help sync test lint types check eval build install clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

sync: ## Install dependencies (uv sync)
	uv sync

test: ## Run the test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

types: ## Type-check with ty
	uv run ty check

check: lint types test ## Lint + type-check + tests (the pre-commit gate)

eval: ## Score citation accuracy against evals/cases.toml (file-F1 + line-F1)
	uv run python evals/run.py

build: ## Build the wheel
	uv build --wheel

install: ## Install the fcx CLI globally (uv tool install)
	uv tool install .

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
