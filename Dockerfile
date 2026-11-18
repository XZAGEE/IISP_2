
# Етап 1: навчання моделі
FROM python:3.11-slim AS trainer

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Встановлення залежностей для навчання
COPY requirements-api.txt requirements-train.txt . 

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements-train.txt

# Копіювання коду та датасету
COPY src/ ./src/

ARG DATASET_DIR=dataset
COPY ${DATASET_DIR}/ ./dataset/

# Параметри навчання
ARG EPOCHS=5
ARG SEED=42

# Навчання та збереження артефактів
RUN python -m src.train \
    --classes yes no up down \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --out artifacts


# Етап 2: запуск API
FROM python:3.11-slim AS runtime

WORKDIR /app

# Системна бібліотека для PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Залежності лише для API
COPY requirements-api.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements-api.txt

# Код API та готові артефакти з першого етапу
COPY src/ ./src/
COPY --from=trainer /app/artifacts/ ./artifacts/

# Налаштування API
ENV MODEL_PATH=/app/artifacts/model.pt
ENV LABELS_PATH=/app/artifacts/labels.json
ENV METRICS_PATH=/app/artifacts/metrics.json
ENV MODEL_VERSION=1.0.0
ENV PORT=8000
ENV LOG_LEVEL=info

EXPOSE 8000

# Перевірка працездатності API
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Запуск від непривілейованого користувача
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]
