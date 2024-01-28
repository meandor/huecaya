import csv
import json
import logging
import os.path
import pathlib
from typing import Any, Optional, Sequence

from experimentation.domain.domain_model import Evaluation, Experiment

_UTF_8 = "utf-8"


def _write_csv(path: str, content: Sequence[Optional[str]]) -> None:
    with open(path, "a", encoding=_UTF_8) as file:
        csv_writer = csv.writer(
            file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL
        )
        for row in content:
            if row:
                row = row.replace("\n", "\\n")
            csv_writer.writerow([row])


def _write_json(path: str, content: Any) -> None:
    with open(path, "a", encoding=_UTF_8) as file:
        json.dump(content, file)
        file.write("\n")


def simple_file_persistor(
    output_path: str,
    experiment: Experiment,
    labels: list[str],
    predicted: list[str],
    evaluation_results: Evaluation,
) -> None:
    serializable_config = {
        **experiment,
        "prompt_template": experiment["prompt_template"].template,
    }
    base_path = os.path.join(output_path, "results", experiment["name"])
    logging.info("Persisting results to: %s", base_path)
    metrics_path = os.path.join(base_path, "metrics.json")
    predicted_path = os.path.join(base_path, "predicted.csv")
    actual_path = os.path.join(base_path, "labels.csv")
    config_path = os.path.join(base_path, "config.json")
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

    _write_json(metrics_path, evaluation_results)
    _write_json(config_path, serializable_config)
    _write_csv(predicted_path, predicted)
    _write_csv(actual_path, labels)
