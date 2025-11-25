.PHONY: lint lint-notebooks format format-notebooks test test-notebooks

lint:
	./scripts/lint.sh

lint-notebooks:
	./scripts/lint_notebooks.sh

format:
	./scripts/clean.sh

format-notebooks:
	./scripts/clean_notebooks.sh

test: lint
	./scripts/test.sh

test-notebooks: lint-notebooks
	./scripts/test_notebooks.sh
