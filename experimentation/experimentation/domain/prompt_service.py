from typing import Callable

from langchain.prompts import PromptTemplate

from experimentation.data.prompts import PromptTemplateParams
from experimentation.domain.domain_model import PromptTemplateFactoryFn


def prompt_template_factory(
    loader: Callable[[str], PromptTemplateParams],
) -> PromptTemplateFactoryFn:
    def _generate(path: str) -> PromptTemplate:
        params = loader(path)
        return PromptTemplate(
            template=params.template, input_variables=params.variables
        )

    return _generate
