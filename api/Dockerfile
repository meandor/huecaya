FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY resources ./resources
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY de_identification_api ./de_identification_api
ENV PYTHONPATH=/usr/src/app

RUN useradd -M -s /bin/bash de_identification_api && chown -R de_identification_api:de_identification_api /usr/src/app
USER de_identification_api

CMD [ "gunicorn", "-b", "0.0.0.0:8000", "-w", "4", "de_identification_api.presentation.endpoints:start_server('./resources/', './resources/mistral-7b-instruct-v0.1.Q5_K_M.gguf')" ]
