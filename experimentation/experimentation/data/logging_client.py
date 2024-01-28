import logging
import os
from contextlib import contextmanager
from typing import Generator

import mlflow
from langchain.prompts.prompt import PromptTemplate

from experimentation.domain.domain_model import Experiment, LoggingClientFn, Evaluation

_EXPERIMENT_PREFIX = "de-identification_"


def _to_prompt(prompt_template: PromptTemplate, sentence: str) -> str:
    return prompt_template.format(sentence=sentence)


def mlflow_logging_client() -> LoggingClientFn:
    @contextmanager
    def _run(experiment: Experiment) -> Generator[None, None, None]:
        logging.info("Setting up experiments in mlflow")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(_EXPERIMENT_PREFIX + experiment["model_name"])

        logging.info("Starting mlflow run")
        mlflow.start_run(run_name=experiment["name"])
        mlflow.log_params(experiment["model_parameters"])
        try:
            yield
        except Exception as error:  # pylint: disable=W0718
            logging.error(
                "Could not finish experiment: %s", experiment["name"], exc_info=error
            )
            mlflow.end_run("FAILED")
            raise
        logging.info("Ending mlflow run")
        mlflow.end_run()

    def _log_inference(
        experiment: Experiment,
        step: int,
        sentence: str,
        answer: str,
        metrics: Evaluation,
    ) -> None:
        logging.info("Logging predictions in mlflow")
        table_data = {
            "inputs": [_to_prompt(experiment["prompt_template"], sentence)],
            "prompt_template": [experiment["prompt_template"].template],
            "outputs": [answer],
        }
        mlflow.log_table(data=table_data, artifact_file="artifacts.json")
        logging.info("Logging metrics in mlflow")
        mlflow.log_metrics(metrics, step=step)

    return _run, _log_inference


def noop_logging_client() -> LoggingClientFn:
    @contextmanager
    def _run(experiment: Experiment) -> Generator[None, None, None]:
        try:
            yield
        except Exception as error:  # pylint: disable=W0718
            logging.error(
                "Could not finish experiment: %s", experiment["name"], exc_info=error
            )
            raise

    def _log_inference(
        _experiment: Experiment,
        _step: int,
        _sentence: str,
        _answer: str,
        _metrics: Evaluation,
    ) -> None:
        pass

    return _run, _log_inference
