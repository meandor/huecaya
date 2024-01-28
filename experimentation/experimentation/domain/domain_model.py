from typing import Callable, Optional, TypedDict, Any, ContextManager

from langchain.prompts import PromptTemplate

PREDICTION_DELIMITER = "###"


class ModelParameters(TypedDict):
    max_length: int
    temperature: float
    context_length: int


class Experiment(TypedDict):
    model_parameters: ModelParameters
    model_name: str
    prompt_template: PromptTemplate
    name: str


class Evaluation(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class ConfusionMatrix(TypedDict):
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class EvaluationWithConfusionMatrix(Evaluation, ConfusionMatrix):
    pass


InferenceFn = Callable[[str], str]
PromptTemplateFactoryFn = Callable[[str], PromptTemplate]
ParameterLoaderFn = Callable[[str], dict[str, Any]]
InferenceFactoryFn = Callable[[Experiment], InferenceFn]
ResultPersistor = Callable[[Experiment, list[str], list[str], Evaluation], None]
EvaluationFn = Callable[[list[str], list[Optional[str]]], Evaluation]
LoggingInferenceFn = Callable[[Experiment, int, str, str, Evaluation], None]
LoggingClientFn = tuple[
    Callable[[Experiment], ContextManager[None]], LoggingInferenceFn
]
