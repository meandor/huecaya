import logging
import os
import time
import requests
from requests.exceptions import HTTPError


_SPLIT_CHARACTER = ","
_TIMEOUT_SECONDS = 3
_MAX_RETRIES = 5
_RETRY_DELAY = 1


def _post_request(resource: str, payload: dict[str, str]) -> list[str]:
    response = requests.post(resource, json=payload, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()

    answer = response.json()["answer"]
    parsed_answer = map(lambda x: x.strip(), answer.split(_SPLIT_CHARACTER))

    return list(parsed_answer)


def send_deidentification_request(endpoint: str, text: str) -> list[str]:  # type: ignore
    resource = os.path.join(endpoint, "prompt")
    payload = {"prompt": text}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _post_request(resource, payload)
        except HTTPError as error:
            logging.warning(f"Attempt {attempt} failed with error: {error}")
            if attempt < _MAX_RETRIES:
                logging.info(f"Retrying in {_RETRY_DELAY} seconds...")
                time.sleep(_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"There was an error calling the endpoint: '{endpoint}'", error
                )
