import re
from copy import deepcopy
from functools import partial, reduce
from typing import Optional

import numpy as np

from experimentation.domain.domain_model import (
    PREDICTION_DELIMITER,
    ConfusionMatrix,
    EvaluationWithConfusionMatrix,
)

_MISSING_PREDICTION = "!"
_MISSING_LABEL = "_"
_NONE_STRING = "none"


def _split_by(character: str, acc: list[str], sentence: Optional[str]) -> list[str]:
    if not sentence:
        return acc + [""]
    return acc + list(map(lambda x: x.strip(), sentence.split(character)))


def _has_entity(line: str) -> bool:
    return " | True | " in line


def _pad(
    original_array: list[str], desired_shape: int, constant_value: str
) -> list[str]:
    return np.array(  # type: ignore
        [
            original_array[i] if i < len(original_array) else constant_value
            for i in range(desired_shape)
        ]
    ).tolist()


def _is_chain_of_thought(sentence: str) -> bool:
    pattern = r"\d+\.\s.+\s\|\s(?:True|False)\s\|\s\w+"
    return bool(re.search(pattern, sentence))


def _sanitize_labels(
    predicted: list[str], labels: list[Optional[str]]
) -> tuple[list[str], list[str]]:
    first = predicted[0]
    if _is_chain_of_thought(first):
        keep_first_prediction = map(
            lambda x: x.split(PREDICTION_DELIMITER)[0], predicted
        )
        lines: list[str] = reduce(partial(_split_by, "\n"), keep_first_prediction, [])
        only_entity_lines = filter(_has_entity, lines)
        split_lines = map(partial(_split_by, " | ", []), only_entity_lines)
        predicted_labels = list(map(lambda x: x[0].split(".")[1].strip(), split_lines))
    else:
        predicted_labels: list[str] = reduce(partial(_split_by, ","), predicted, [])  # type: ignore
    actual_labels: list[str] = reduce(partial(_split_by, ","), labels, [])
    return predicted_labels, actual_labels


def _remove_titles(sequence: list[str]) -> list[str]:
    return list(
        map(
            lambda x: x.lower()
            .replace("dr", "")
            .replace("prof", "")
            .replace("med", "")
            .replace(".", "")
            .replace(_NONE_STRING, "")
            .replace("        ", " ")
            .replace("      ", " ")
            .replace("    ", " ")
            .replace("  ", " ")
            .strip(),
            sequence,
        )
    )


def _pad_labels(predicted: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    actual_length = len(labels)
    predicted_length = len(predicted)
    actual_labels = deepcopy(labels)
    predicted_labels = deepcopy(predicted)
    if actual_length < predicted_length:
        actual_labels = _pad(labels, predicted_length, _MISSING_LABEL)
    elif actual_length > predicted_length:
        predicted_labels = _pad(predicted, actual_length, _MISSING_PREDICTION)
    return predicted_labels, actual_labels


def _extract_names_from_chain_of_thought(prediction: str) -> str:
    lines = prediction.strip().split("\n")
    lines_without_empty = filter(lambda x: x, lines)
    name_lines = filter(_has_entity, lines_without_empty)
    names = map(lambda x: x.split("|", 1)[0].strip().split(" ", 1)[-1], name_lines)
    return ", ".join(names)


def _select_first_prediction(raw_prediction: str) -> str:
    return raw_prediction.split(PREDICTION_DELIMITER)[0].strip()


def _calculate_confusion_matrix(
    initial: ConfusionMatrix,
    predicted_tokens: set[str],
    actual_tokens: set[str],
    actual_label: str,
) -> ConfusionMatrix:
    confusion_matrix = deepcopy(initial)
    for predicted_token in predicted_tokens:
        if predicted_token != "" and predicted_token in actual_tokens:
            confusion_matrix["true_positives"] += 1
        elif predicted_token != "" and predicted_token not in actual_tokens:
            confusion_matrix["false_positives"] += 1
        elif predicted_token == "" and predicted_token == actual_label:
            confusion_matrix["true_negatives"] += 1
    for actual_token in actual_tokens:
        if actual_token != "" and actual_token not in predicted_tokens:
            confusion_matrix["false_negatives"] += 1
    return confusion_matrix


def _calculate_metrics(
    confusion_matrix: ConfusionMatrix,
) -> EvaluationWithConfusionMatrix:
    positives = confusion_matrix["true_positives"] + confusion_matrix["false_negatives"]
    negatives = confusion_matrix["true_negatives"] + confusion_matrix["false_positives"]
    return {
        "accuracy": 0
        if (positives + negatives) == 0
        else (confusion_matrix["true_positives"] + confusion_matrix["true_negatives"])
        / (positives + negatives),
        "precision": 0
        if (confusion_matrix["true_positives"] + confusion_matrix["false_positives"])
        == 0
        else confusion_matrix["true_positives"]
        / (confusion_matrix["true_positives"] + confusion_matrix["false_positives"]),
        "recall": 0
        if positives == 0
        else confusion_matrix["true_positives"] / positives,
        "f1_score": 0
        if (
            2 * confusion_matrix["true_positives"]
            + confusion_matrix["false_positives"]
            + confusion_matrix["false_negatives"]
        )
        == 0
        else (2 * confusion_matrix["true_positives"])
        / (
            2 * confusion_matrix["true_positives"]
            + confusion_matrix["false_positives"]
            + confusion_matrix["false_negatives"]
        ),
        **confusion_matrix,
    }


def _extract_names(raw_prediction: str) -> str:
    return raw_prediction.split("\n")[0].replace("'", "").strip()


def _remove_duplicates(values: list[str]) -> set[str]:
    unique = set(values)
    if len(unique) > 1:
        return set(filter(lambda x: x, unique))
    return unique


def evaluate_tokens(
    predicted: list[str], labels: list[Optional[str]]
) -> EvaluationWithConfusionMatrix:
    confusion_matrix: ConfusionMatrix = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    for index, raw_prediction in enumerate(predicted):
        actual_label = labels[index] or ""
        actual_label_without_titles = _remove_titles(actual_label.split(","))
        actual_tokens = reduce(partial(_split_by, " "), actual_label_without_titles, [])  # type: ignore
        prediction = _select_first_prediction(raw_prediction)
        if _is_chain_of_thought(prediction):
            prediction_names = _extract_names_from_chain_of_thought(raw_prediction)
        else:
            prediction_names = _extract_names(raw_prediction)
        prediction_without_titles = _remove_titles(prediction_names.split(","))
        predicted_tokens: list[str] = reduce(
            partial(_split_by, " "), prediction_without_titles, []
        )

        confusion_matrix = _calculate_confusion_matrix(
            confusion_matrix,
            _remove_duplicates(predicted_tokens),
            _remove_duplicates(actual_tokens),
            actual_label,
        )

    return _calculate_metrics(confusion_matrix)
