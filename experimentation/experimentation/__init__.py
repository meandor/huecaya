import logging
import os
import sys

_LOCAL_LOGGING_FILE = "console.log"

logging.basicConfig(
    format="%(asctime)s %(levelname)s	%(message)s "
    "[%(process)d] %(module)s %(filename)s %(funcName)s",
    level=os.environ.get("LOGLEVEL", "INFO"),
    handlers=[
        logging.FileHandler(_LOCAL_LOGGING_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
