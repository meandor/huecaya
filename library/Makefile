SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

SRC_PATH := "./de_identification_library"

.PHONY: help
help: ## Shows help for targets with help text
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: install
install: ## Installs all dependencies
	poetry install

.PHONY: requirements.txt
requirements.txt: ## Creates requirements.txt file
	poetry export -f requirements.txt --output requirements.txt

.PHONY: lint
lint: ## Checks code formatting and style
	poetry run ruff check $(SRC_PATH) ./tests

.PHONY: type-check
type-check: ## Checks types
	poetry run mypy --ignore-missing-imports --strict --allow-subclassing-any \
		$(SRC_PATH)

.PHONY: auto-format
auto-format: ## Automatically formats the code
	poetry run ruff format $(SRC_PATH) ./tests

.PHONY: unit-test
unit-test: ## Runs all unit tests
	HUECAYA_ENDPOINT="http://localhost:4545" poetry run pytest -v -s ./tests/unit

.PHONY: integration-test
integration-test: ## Runs all integration tests
	docker pull bbyars/mountebank:2.9.1
	docker rm -f huecaya_mock_backend
	docker run -d -p 2525:2525 -p 4545:4545 -p 5555:5555 --name huecaya_mock_backend -v ./tests/integration/resources:/app/data bbyars/mountebank:2.9.1 start --datadir /app/data
	sleep 4
	HUECAYA_ENDPOINT="http://localhost:4545" poetry run pytest -v -s ./tests/integration
	docker rm -f huecaya_mock_backend

.PHONY: test
test: unit-test integration-test ## Runs all tests

.PHONY: check
check: auto-format lint type-check test ## Runs all code checks and tests
