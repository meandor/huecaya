from functools import reduce
from de_identification_library import client
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import os

_ENDPOINT = os.environ["HUECAYA_ENDPOINT"]
_REDACTION_PATTERN = "xxx"


def _redact(acc: str, element: str) -> str:
    return acc.replace(element, _REDACTION_PATTERN).replace(
        element.lower(), _REDACTION_PATTERN
    )


def de_identify_fn(endpoint: str, text: str) -> str:
    response = client.send_deidentification_request(endpoint, text)

    return reduce(_redact, response, text)


@udf(returnType=StringType())
def de_identify(text: str) -> str:
    return de_identify_fn(_ENDPOINT, text)
