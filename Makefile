.PHONY: help data test clean

help:
	@echo "Commands:"
	@echo "  make data    - Download data"
	@echo "  make test    - Run tests"
	@echo "  make clean   - Clean cache"

data:
	@python3 -m src.data.ingestion

test:
	@pytest tests/ -v

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache
	@rm -rf data/raw/*
