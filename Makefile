PYTHON ?= python3

.PHONY: test hooks

test:
	$(PYTHON) -m unittest discover -v

hooks:
	./scripts/install_hooks.sh
