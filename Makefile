.PHONY: help test clean

help:
	@echo "Available commands:"
	@echo "  make test    - Run all tests"
	@echo "  make clean   - Clean cache files"

test:
	PYTHONPATH=$$PWD pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
