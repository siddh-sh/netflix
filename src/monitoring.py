"""Prometheus metrics exposed by the Streamlit app on a separate port."""
import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

PAGE_VIEWS = Counter("netflix_app_page_views_total", "Streamlit page views", ["page"])
PREDICTIONS = Counter("netflix_model_predictions_total", "Model predictions", ["label"])
PREDICTION_LATENCY = Histogram("netflix_model_prediction_latency_seconds", "Prediction latency in seconds")
PREDICTION_CONFIDENCE = Histogram(
    "netflix_model_prediction_confidence", "Predicted class probability",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_ACCURACY = Gauge("netflix_model_accuracy", "Hold-out accuracy of the loaded model")
MODEL_F1 = Gauge("netflix_model_f1_macro", "Hold-out macro F1 of the loaded model")
DATASET_ROWS = Gauge("netflix_dataset_rows", "Rows in the loaded dataset")

_started = False


def start_metrics_server() -> None:
    global _started
    if not _started:
        start_http_server(METRICS_PORT)
        _started = True
