"""Container entrypoint: expose Prometheus metrics immediately, then start Streamlit.

Streamlit only executes the app script when a browser connects, so starting the
metrics server here keeps the Prometheus target up from container start.
"""
import sys

from streamlit.web import cli

from src import monitoring

if __name__ == "__main__":
    monitoring.start_metrics_server()
    sys.argv = [
        "streamlit", "run", "app/streamlit_app.py",
        "--server.port=8501", "--server.address=0.0.0.0",
    ]
    sys.exit(cli.main())
