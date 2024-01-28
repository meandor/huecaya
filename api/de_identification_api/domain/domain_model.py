from typing import TypedDict, Callable


class ModelParameters(TypedDict):
    max_length: int
    temperature: float
    context_length: int


ParametersLoader = Callable[[str], ModelParameters]
InferenceFn = Callable[[str], str]
