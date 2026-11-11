FROM python:3.12-slim

# LightGBM needs the OpenMP runtime on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first, so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements-api.txt

COPY src/ src/
COPY models/ models/

# Don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]