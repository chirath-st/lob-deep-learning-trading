.PHONY: setup data train evaluate test lint clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up Python environment and install dependencies
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Run: source .venv/bin/activate"

data: ## Download and preprocess FI-2010 dataset
	python scripts/download_data.py
	python scripts/preprocess_data.py

train: ## Train DeepLOB (usage: make train CONFIG=deeplob_fi2010)
	python scripts/train.py --config configs/$(or $(CONFIG),deeplob_fi2010).yaml

evaluate: ## Evaluate a trained model (usage: make evaluate EXP=exp001)
	python scripts/evaluate.py --experiment $(EXP)

test: ## Run all tests
	pytest tests/ -v --tb=short

lint: ## Lint and format code
	ruff check src/ tests/ scripts/
	black src/ tests/ scripts/

format: ## Auto-format code
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean: ## Remove cached/generated files (NOT raw data)
	rm -rf data/processed/*
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

explore: ## Launch Jupyter for data exploration
	jupyter notebook notebooks/
