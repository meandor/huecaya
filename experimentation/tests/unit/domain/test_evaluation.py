from experimentation.domain import evaluation


def test_should_return_scores_tokens_with_1_negative():
    predicted = ["", "foo"]
    labels = [None, "foo"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 1,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_return_scores_tokens_all_positive():
    predicted = ["a, foobar"]
    labels = ["dr. a foobar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 0,
        "true_positives": 2,
    }

    assert expected == actual


def test_should_return_scores_tokens_1_tp_1_fn():
    predicted = ["a foo"]
    labels = ["dr. a foo, bar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 2 / 3,
        "precision": 1.0,
        "recall": 2 / 3,
        "f1_score": 0.8,
        "false_negatives": 1,
        "false_positives": 0,
        "true_negatives": 0,
        "true_positives": 2,
    }

    assert expected == actual


def test_should_return_scores_tokens_false_positives():
    predicted = [""]
    labels = ["dr. a foo, bar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "false_negatives": 3,
        "false_positives": 0,
        "precision": 0,
        "recall": 0.0,
        "true_negatives": 0,
        "true_positives": 0,
    }

    assert expected == actual


def test_should_return_scores_tokens_one_tp():
    predicted = ["a"]
    labels = ["dr. a foo, bar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1 / 3,
        "f1_score": 0.5,
        "false_negatives": 2,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1 / 3,
        "true_negatives": 0,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_return_scores_token_with_cleaned_strings():
    predicted = [" None\n", " a \n "]
    labels = ["", "a"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 1,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_ignore_order_and_titles_in_token():
    predicted = [
        "Bob Bobo, Toto",
        "",
        "",
        "",
        "Coco Wewe, Sese Wawa",
        "Iaia Steste, Hehe Dede",
    ]
    labels = [
        "Toto, Bob Bobo",
        "",
        "",
        "",
        "Prof. Dr. Coco      Wewe, Dr. med. Sese Wawa",
        "Dr. med. Iaia Steste, Hehe Dede",
    ]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 3,
        "true_positives": 11,
    }

    assert expected == actual


def test_should_evaluate_prompt_ner_with_tokens():
    prediction = """
    1. foobar | False | something foobar
    2. Toto | True | something else
    """.strip()
    predicted = [prediction]
    labels = ["Toto"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 0,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_return_false_negative_and_false_positive_for_wrong_names():
    predicted = ["foo, bar"]
    labels = ["foobar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "false_negatives": 1,
        "false_positives": 2,
        "precision": 0.0,
        "recall": 0.0,
        "true_negatives": 0,
        "true_positives": 0,
    }

    assert expected == actual


def test_should_evaluate_only_first_separated_by_delimiter():
    predicted = [" foobar\n\n###\n\nsomething else"]
    labels = ["foobar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 0,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_evaluate_only_first_separated_by_new_line():
    predicted = [" foobar\n\nsomething else"]
    labels = ["foobar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 0,
        "true_positives": 1,
    }

    assert expected == actual


def test_should_evaluate_unique_tokens():
    predicted = [" foobar, foobar, dr."]
    labels = [""]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "false_negatives": 0,
        "false_positives": 1,
        "precision": 0.0,
        "recall": 0.0,
        "true_negatives": 0,
        "true_positives": 0,
    }

    assert expected == actual


def test_should_ignore_single_quote_for_tokens():
    predicted = [" 'foobar'"]
    labels = ["foobar"]

    actual = evaluation.evaluate_tokens(predicted, labels)
    expected = {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "false_negatives": 0,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "true_negatives": 0,
        "true_positives": 1,
    }

    assert expected == actual
