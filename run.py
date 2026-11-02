"""Container entrypoint: expose Prometheus metrics immediately, then start Streamlit.

Streamlit only executes the app script when a browser connects, so starting the
metrics server here keeps the Prometheus target up from container start.
"""
import json
import sys

from streamlit.web import cli

from src import monitoring
from src.data import load_clean
from src.train import METRICS_PATH, MODEL_PATH, train

if __name__ == "__main__":
    monitoring.start_metrics_server()
    if not MODEL_PATH.exists():
        train()
    model_metrics = json.loads(METRICS_PATH.read_text())
    monitoring.MODEL_ACCURACY.set(model_metrics["accuracy"])
    monitoring.MODEL_F1.set(model_metrics["f1_macro"])
    monitoring.DATASET_ROWS.set(len(load_clean()))
    sys.argv = [
        "streamlit", "run", "app/streamlit_app.py",
        "--server.port=8501", "--server.address=0.0.0.0",
    ]
    sys.exit(cli.main())
