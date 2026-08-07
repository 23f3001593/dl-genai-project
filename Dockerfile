# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/warm_cache.py ./src/warm_cache.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python src/warm_cache.py

COPY app.py .
COPY src/ ./src/

ENV PORT=8080
ENV HF_HUB_OFFLINE=1
EXPOSE 8080

CMD ["python", "app.py"]