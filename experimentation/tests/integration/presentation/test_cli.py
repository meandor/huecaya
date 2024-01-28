from experimentation.presentation import cli


def test_should_execute():
    experiment_path = "tests/integration/resources/simple-experiment"
    dataset_path = "tests/integration/resources/"
    model_path = "tests/integration/resources/model.gguf"
    use_mlflow = None
    arguments = [
        "--experiment-path",
        experiment_path,
        "--dataset-path",
        dataset_path,
        "--model-path",
        model_path,
        "--use-mlflow",
        use_mlflow,
    ]
    actual = cli.main_call(arguments, standalone_mode=False)
    expected = 0

    assert expected == actual
