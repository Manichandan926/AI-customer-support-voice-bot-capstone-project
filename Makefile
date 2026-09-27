PYTHON ?= python3

.PHONY: test hooks

test:
	$(PYTHON) -m pytest tests -v

hooks:
	./scripts/install_hooks.sh
