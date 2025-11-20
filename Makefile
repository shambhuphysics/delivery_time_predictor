.PHONY: help data clean-data split test clean pipeline

help:
	@echo "Commands:"
	@echo "  make pipeline    - Run full data pipeline"
	@echo "  make data        - Download data"
	@echo "  make clean-data  - Clean data"
	@echo "  make split       - Split data"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean cache"

data:
	@python3 -m src.data.ingestion

clean-data:
	@python3 -m src.data.cleaning

split:
	@python3 -m src.data.splitting

pipeline: data clean-data split
	@echo "✓ Pipeline complete"

test:
	@pytest tests/ -v

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache
