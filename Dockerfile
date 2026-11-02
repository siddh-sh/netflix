FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METRICS_PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY VERSION .
COPY data/ data/
COPY src/ src/
COPY app/ app/
COPY run.py .
COPY .streamlit/ .streamlit/

# Train the model at build time so the image ships with a versioned artifact
RUN python -m src.train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "run.py"]
