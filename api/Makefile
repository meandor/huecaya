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

SRC_PATH := "./de_identification_api"

.PHONY: help
help: ## Shows help for targets with help text
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: install
install: ## Installs all dependencies
	rm -rf resources/mistral-7b-instruct-v0.1.Q5_K_M.gguf
	wget -O resources/mistral-7b-instruct-v0.1.Q5_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
	poetry install

.PHONY: requirements.txt
requirements.txt: ## Creates requirements.txt file
	poetry export -f requirements.txt --output requirements.txt

.PHONY: lint
lint: ## Checks code formatting and style
	poetry run ruff check $(SRC_PATH)

.PHONY: type-check
type-check: ## Checks types
	poetry run mypy --ignore-missing-imports --strict --allow-subclassing-any \
		$(SRC_PATH)

.PHONY: auto-format
auto-format: ## Automatically formats the code
	poetry run ruff format $(SRC_PATH)

.PHONY: check
check: auto-format lint type-check ## Runs all code checks and tests

.PHONY: run
run: ## Runs a flask server
	NUMEXPR_MAX_THREADS=64 poetry run flask --app "de_identification_api.presentation.endpoints:start_server('./resources/', './resources/mistral-7b-instruct-v0.1.Q5_K_M.gguf')" run -p 5001
