import logging
from typing import Any, Optional

from flask import Flask, Response, jsonify, make_response, request

from de_identification_api.data import loaders
from de_identification_api.domain.inference import generate_inference_fn


def _is_valid_prompt(data: Optional[Any]) -> bool:
    return bool(data and data.get("prompt"))


def start_server(config_path: str, model_path: str) -> Flask:
    logging.info("Starting server")
    app = Flask(__name__)
    logging.info("Loading model")
    prompt_template = loaders.load_prompt_template(config_path)
    parameters = loaders.load_parameters(config_path)
    inference_fn = generate_inference_fn(parameters, prompt_template, model_path)

    @app.route("/health")
    def ping() -> Response:
        return jsonify({"status": "ok"})

    @app.route("/prompt", methods=["POST"])
    def create_prompt() -> Response:
        data = request.get_json()
        if not _is_valid_prompt(data):
            response_body = {"error": "invalid request"}
            status_code = 400
            response = make_response(jsonify(response_body), status_code)
            return response

        prompt = data["prompt"]
        logging.info("Processing prompt")
        answer = inference_fn(prompt)
        return jsonify({"answer": answer})

    return app
