FROM python:3.11-slim

LABEL maintainer="IUE Big Data"
LABEL description="Pipeline ML reproducible con datos astronómicos SDSS"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sdss_sample.csv .
COPY src/ ./src/
COPY tests/ ./tests/

RUN mkdir -p outputs

CMD ["sh", "-c", "python tests/test_dataset.py && cd src && python main.py"]
