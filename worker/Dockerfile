FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG MODEL_ID
ENV MODEL_ID=${MODEL_ID} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY worker/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY worker/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
