import json
import os.path
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset

_PERSON_ENTITY_TAG = "PER"
_WORDS_COLUMN = "words"
_NER_COLUMN = "ner"
INPUT_PROMPT_COLUMN = "input"
OUTPUT_PROMPT_COLUMN = "output"
_PROMPT_COLUMN = "prompt"
_CSV_INPUT_COLUMN = "text"
_CSV_LABEL_COLUMN = "labels"


def _to_example_prompt(acc: str, example: dict[str, Any]) -> str:
    prompt = (
        f"Q: {example[INPUT_PROMPT_COLUMN]}\n" f"A: {example[OUTPUT_PROMPT_COLUMN]}\n"
    )
    return acc + prompt


def _to_inference(data: dict[str, Any]) -> dict[str, Any]:
    input_sentences = " ".join(data[_WORDS_COLUMN])
    data[_PROMPT_COLUMN] = f"Q: {input_sentences}\nA: "
    return data


def load_csv_dataset(dataset_path: str) -> Dataset:
    test_dataset = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    return Dataset.from_pandas(test_dataset)


def csv_dataset_to_example(csv_dataset: dict[str, Any]) -> dict[str, Any]:
    csv_dataset[INPUT_PROMPT_COLUMN] = csv_dataset[_CSV_INPUT_COLUMN]
    csv_dataset[OUTPUT_PROMPT_COLUMN] = csv_dataset[_CSV_LABEL_COLUMN]
    return csv_dataset


@dataclass
class PromptTemplateParams:
    template: str
    variables: list[str]


def prompt_template_json_loader(path: str) -> PromptTemplateParams:
    prompt_template_file = os.path.join(path, "prompt_template.json")
    with open(prompt_template_file, "r", encoding="utf-8") as file:
        content = json.load(file)
        return PromptTemplateParams(
            template=content["template"], variables=content["variables"]
        )
