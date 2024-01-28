from unittest.mock import MagicMock, ANY

from langchain.prompts import PromptTemplate

from experimentation.domain.experiment_service import create_grid, load_experiments


def test_should_return_simple_grid():
    params = {"foo": [1], "bar": "foobar", "x": 1337}

    actual = create_grid(params)
    expected = [{"foo": 1, "bar": "foobar", "x": 1337}]

    assert expected == actual


def test_should_return_grid():
    params = {"foo": range(3), "bar": "foobar", "foobar": range(2)}

    actual = create_grid(params)
    expected = [
        {"foo": 0, "bar": "foobar", "foobar": 0},
        {"foo": 0, "bar": "foobar", "foobar": 1},
        {"foo": 1, "bar": "foobar", "foobar": 0},
        {"foo": 1, "bar": "foobar", "foobar": 1},
        {"foo": 2, "bar": "foobar", "foobar": 0},
        {"foo": 2, "bar": "foobar", "foobar": 1},
    ]

    assert expected == actual


def test_should_return_one_experiment():
    prompt_template_loader_mock = MagicMock()
    prompt_template_loader_mock.return_value = PromptTemplate(
        template="foo {bar}", input_variables=["bar"]
    )
    parameter_loader_mock = MagicMock()
    parameter_loader_mock.return_value = {"foo": 1}
    experiment_path = "/tmp/foobar"
    model_path = "/tmp/models/foobar.gguf"

    actual = load_experiments(
        prompt_template_loader_mock, parameter_loader_mock, model_path, experiment_path
    )
    expected = [
        {
            "model_name": "foobar",
            "model_parameters": {"foo": 1},
            "name": ANY,
            "prompt_template": PromptTemplate(
                template="foo {bar}", input_variables=["bar"]
            ),
        }
    ]

    assert expected == actual


def test_should_return_n_experiments():
    prompt_template_loader_mock = MagicMock()
    prompt_template_loader_mock.return_value = PromptTemplate(
        template="{foobar}", input_variables=["foobar"]
    )
    parameter_loader_mock = MagicMock()
    parameter_loader_mock.return_value = {"foo": [1, 2, 3]}
    experiment_path = "/tmp/foobar"
    model_path = "/tmp/models/foobar-0.1-v-3.gguf"

    actual = load_experiments(
        prompt_template_loader_mock, parameter_loader_mock, model_path, experiment_path
    )
    expected = [
        {
            "model_name": "foobar-0_1-v-3",
            "model_parameters": {"foo": 1},
            "name": ANY,
            "prompt_template": PromptTemplate(
                template="{foobar}", input_variables=["foobar"]
            ),
        },
        {
            "model_name": "foobar-0_1-v-3",
            "model_parameters": {"foo": 2},
            "name": ANY,
            "prompt_template": PromptTemplate(
                template="{foobar}", input_variables=["foobar"]
            ),
        },
        {
            "model_name": "foobar-0_1-v-3",
            "model_parameters": {"foo": 3},
            "name": ANY,
            "prompt_template": PromptTemplate(
                template="{foobar}", input_variables=["foobar"]
            ),
        },
    ]

    assert expected == actual
