from textwrap import dedent

from experimentation.data import prompts
from experimentation.data.prompts import PromptTemplateParams


def test_should_return_simple_prompt_template_params():
    actual = prompts.prompt_template_json_loader(
        "tests/unit/resources/simple-experiment"
    )
    expected = PromptTemplateParams(template="foo: {bar}", variables=["bar"])

    assert expected == actual


def test_should_return_complex_prompt_template_params():
    actual = prompts.prompt_template_json_loader(
        "tests/unit/resources/complex-experiment"
    )
    expected_template = dedent(
        """
    foo: {bar}
    
    answer: foobar {answer}
    """
    ).lstrip()
    expected = PromptTemplateParams(
        template=expected_template, variables=["bar", "answer"]
    )

    assert expected == actual
