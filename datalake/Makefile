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

DATALAKE ?= "../datalake/mlflow"

.PHONY: help
help: ## Shows help for targets with help text
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: llama2-model
llama2-model: ## Downloads llama 2 model from huggingface
	wget -P ./models/ https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf

.PHONY: falcon-model
falcon-model: ## Downloads falcon model from huggingface
	wget -P ./models/ https://huggingface.co/maddes8cht/tiiuae-falcon-7b-gguf/resolve/main/tiiuae-falcon-7b-Q5_K_M.gguf

.PHONY: falcon-instruct-model
falcon-instruct-model: ## Downloads falcon instruct model from huggingface
	wget -P ./models/ https://huggingface.co/maddes8cht/tiiuae-falcon-7b-instruct-gguf/resolve/main/tiiuae-falcon-7b-instruct-Q5_K_M.gguf

.PHONY: mistral-model
mistral-model: ## Downloads mistral model from huggingface
	wget -P ./models/ https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q5_K_M.gguf

.PHONY: mistral-instruct-model
mistral-instruct-model: ## Downloads mistral instruct model from huggingface
	wget -P ./models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf

.PHONY: models
models: llama2-model falcon-model falcon-instruct-model mistral-model mistral-instruct-model ## Downloads all models from huggingface
