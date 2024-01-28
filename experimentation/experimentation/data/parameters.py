import os
import json
from typing import Any


def json_file_loader(path: str) -> dict[str, Any]:
    parameters_file_path = os.path.join(path, "parameters.json")
    with open(parameters_file_path, "r", encoding="utf-8") as file:
        return json.load(file)  # type: ignore
