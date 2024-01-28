import logging

from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp

from experimentation.domain.domain_model import (
    PREDICTION_DELIMITER,
    InferenceFn,
    Experiment,
)


def generate_inference_fn(model_path: str, experiment: Experiment) -> InferenceFn:
    logging.info("Loading model from: %s", model_path)
    llm = LlamaCpp(
        model_path=model_path,
        echo=False,
        max_tokens=experiment["model_parameters"]["max_length"],
        temperature=experiment["model_parameters"]["temperature"],
        n_ctx=experiment["model_parameters"]["context_length"],
        stop=[PREDICTION_DELIMITER],
    )
    logging.info("Building langchain chain")
    llm_chain = LLMChain(prompt=experiment["prompt_template"], llm=llm)
    logging.info("Done building inference fn")

    def _safe_inference_fn(prompt: str) -> str:
        try:
            return llm_chain.run(prompt)  # type: ignore
        except ValueError as exception:
            logging.warning(
                "Error when applying inference on: %s", prompt, exc_info=exception
            )
            return ""

    return _safe_inference_fn
