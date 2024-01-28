import hashlib
import json
import logging
from functools import partial
from itertools import product
from typing import Any

from datasets import Dataset
from langchain.prompts import PromptTemplate

from experimentation.data.prompts import INPUT_PROMPT_COLUMN, OUTPUT_PROMPT_COLUMN
from experimentation.domain.domain_model import (
    EvaluationFn,
    InferenceFactoryFn,
    InferenceFn,
    Experiment,
    ModelParameters,
    PromptTemplateFactoryFn,
    ResultPersistor,
    ParameterLoaderFn,
    LoggingClientFn,
    LoggingInferenceFn,
)


def _is_single_value(value: Any) -> bool:
    return isinstance(value, (str, int, float))


def _concat_permutations(keys: list[str], permutation: list[Any]) -> dict[str, Any]:
    return dict(zip(keys, permutation))


def _merge(this: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    return {**this, **other}


def create_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    keys = filter(lambda x: not _is_single_value(params[x]), params.keys())
    values = filter(lambda x: not _is_single_value(x), params.values())
    value_products = list(product(*values))
    n_keys = list(keys) * len(value_products)
    grid = map(partial(_concat_permutations, n_keys), value_products)
    keys_single = filter(lambda x: _is_single_value(params[x]), params.keys())
    values_single = filter(_is_single_value, params.values())
    grid_single = dict(zip(keys_single, values_single))
    return list(map(partial(_merge, grid_single), grid))


def _to_name(model_parameters: ModelParameters, prompt_template: PromptTemplate) -> str:
    serializable_config = {
        **model_parameters,
        "prompt_template": prompt_template.template,
    }
    config_hash = hashlib.md5(json.dumps(serializable_config).encode(encoding="utf-8"))
    return config_hash.hexdigest()


def _to_model_name(model_path: str) -> str:
    model = model_path.replace(".gguf", "").split("/")[-1]
    return model.replace(".", "_")


def load_experiments(
    prompt_factory: PromptTemplateFactoryFn,
    parameter_loader: ParameterLoaderFn,
    model_path: str,
    experiment_path: str,
) -> list[Experiment]:
    parameters = parameter_loader(experiment_path)
    grid = create_grid(parameters)
    prompt_template = prompt_factory(experiment_path)
    experiments = []
    for model_parameters in grid:
        experiments += [
            {
                "model_name": _to_model_name(model_path),
                "model_parameters": model_parameters,
                "prompt_template": prompt_template,
                "name": _to_name(model_parameters, prompt_template),  # type: ignore
            }
        ]
    return experiments  # type: ignore


def _run_experiment(
    inference_fn: InferenceFn,
    result_persistor: ResultPersistor,
    evaluation_fn: EvaluationFn,
    log_inference: LoggingInferenceFn,
    dataset: Dataset,
    experiment: Experiment,
) -> None:
    predicted = []
    total_sentences = len(dataset[INPUT_PROMPT_COLUMN])
    for sentence_index, sentence in enumerate(dataset[INPUT_PROMPT_COLUMN]):
        answer = inference_fn(sentence)
        predicted.append(answer)
        logging.info("Calculating current evaluation metrics")
        evaluation_results = evaluation_fn(
            predicted, dataset[OUTPUT_PROMPT_COLUMN][: sentence_index + 1]
        )
        log_inference(experiment, sentence_index, sentence, answer, evaluation_results)
        logging.info("Persisting results")
        result_persistor(
            experiment,
            [dataset[OUTPUT_PROMPT_COLUMN][sentence_index]],
            [answer],
            evaluation_results,
        )
        logging.info(
            "Done running inference %s / %s", sentence_index + 1, total_sentences
        )


def run_experiments(
    inference_factory_fn: InferenceFactoryFn,
    result_persistor: ResultPersistor,
    evaluation_fn: EvaluationFn,
    dataset: Dataset,
    logging_client: LoggingClientFn,
    experiments: list[Experiment],
) -> None:
    for experiment in experiments:
        with logging_client[0](experiment):
            logging.info("Compiling inference fn")
            inference_fn = inference_factory_fn(experiment)
            logging.info("Starting experiment")
            _run_experiment(
                inference_fn,
                result_persistor,
                evaluation_fn,
                logging_client[1],
                dataset,
                experiment,
            )
            logging.info("Done with experiment")
