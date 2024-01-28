from unittest.mock import MagicMock

from langchain.prompts import PromptTemplate

from experimentation.data.prompts import PromptTemplateParams
from experimentation.domain import prompt_service


def test_should_return_a_prompt_template():
    load_template_params_mock = MagicMock()
    load_template_params_mock.return_value = PromptTemplateParams(
        template="{question}", variables=["question"]
    )
    testee = prompt_service.prompt_template_factory(load_template_params_mock)

    actual = testee("/foo/bar.json")
    expected = PromptTemplate(template="{question}", input_variables=["question"])

    assert expected == actual


def test_should_return_another_prompt_template():
    load_template_params_mock = MagicMock()
    load_template_params_mock.return_value = PromptTemplateParams(
        template="a {foobar}", variables=["foobar"]
    )
    testee = prompt_service.prompt_template_factory(load_template_params_mock)

    actual = testee("/foo/bar.json")
    expected = PromptTemplate(template="a {foobar}", input_variables=["foobar"])

    assert expected == actual
