.PHONY: test
test:
	PYTHONPATH=$$PWD pytest tests/ -v
