FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY app ./app

RUN pip install --upgrade pip && pip install -e ".[dev,boost]"

EXPOSE 8000 8501

CMD ["uvicorn", "materials_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
