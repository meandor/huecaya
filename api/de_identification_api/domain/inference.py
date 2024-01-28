import logging

from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp

from de_identification_api.domain.domain_model import InferenceFn, ModelParameters

_PREDICTION_DELIMITER = "###"


def generate_inference_fn(
    parameters: ModelParameters, prompt_template: PromptTemplate, model_path: str
) -> InferenceFn:
    logging.info("Loading model from: %s", model_path)
    llm = LlamaCpp(  # type: ignore
        model_path=model_path,
        echo=False,
        max_tokens=parameters["max_length"],
        temperature=parameters["temperature"],
        n_ctx=parameters["context_length"],
        stop=[_PREDICTION_DELIMITER],
    )
    logging.info("Building langchain chain")
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
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
