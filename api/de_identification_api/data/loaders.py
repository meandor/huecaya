import os.path
from typing import Any

from langchain.prompts import PromptTemplate
import json
from de_identification_api.domain.domain_model import ModelParameters


def _load_json_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)  # type: ignore


def load_parameters(path: str) -> ModelParameters:
    return _load_json_file(os.path.join(path, "parameters.json"))  # type: ignore


def load_prompt_template(path: str) -> PromptTemplate:
    prompt_template = _load_json_file(os.path.join(path, "prompt_template.json"))
    return PromptTemplate(
        template=prompt_template["template"],
        input_variables=prompt_template["variables"],
    )
