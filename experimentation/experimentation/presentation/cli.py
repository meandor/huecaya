import logging
import sys
from functools import partial
from typing import Optional

import click

from experimentation.data import prompts, parameters

from experimentation.data.results_persistence import simple_file_persistor
from experimentation.data.logging_client import (
    mlflow_logging_client,
    noop_logging_client,
)
from experimentation.domain import (
    evaluation,
    experiment_service,
    inference_model,
    prompt_service,
)


@click.command()
@click.option(
    "--experiment-path",
    help="Path to the experiment folder",
)
@click.option(
    "--dataset-path",
    help="Path to the experiment folder",
)
@click.option("--model-path", help="Path to an existing model")
@click.option("--use-mlflow", help="Optional if you want to use mlflow", required=False)
def main_call(
    experiment_path: str,
    dataset_path: str,
    model_path: str,
    use_mlflow: Optional[str],
) -> int:
    logging.info("Start loading experiments")
    prompt_factory = prompt_service.prompt_template_factory(
        prompts.prompt_template_json_loader
    )
    experiments = experiment_service.load_experiments(
        prompt_factory, parameters.json_file_loader, model_path, experiment_path
    )
    logging.info("Found %s experiments", len(experiments))

    logging.info("Loading test dataset: %s", dataset_path)
    inference = prompts.load_csv_dataset(dataset_path)
    test_dataset = inference.map(prompts.csv_dataset_to_example)

    logging.info("Start running experiments")
    inference_fn = partial(inference_model.generate_inference_fn, model_path)
    result_persistor = partial(simple_file_persistor, experiment_path)
    if bool(use_mlflow):
        logging_client = mlflow_logging_client()
    else:
        logging_client = noop_logging_client()
    experiment_service.run_experiments(
        inference_fn,
        result_persistor,
        evaluation.evaluate_tokens,
        test_dataset,
        logging_client,
        experiments,
    )
    logging.info("Done running experiments")
    return 0


# pylint: disable=E1120
if __name__ == "__main__":
    sys.exit(main_call())
