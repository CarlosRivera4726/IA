"""Punto de entrada principal para levantar el servidor de inferencia SentinelCV."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))

from backend_ia.presentation.http_server import run_server

if __name__ == "__main__":
    port = int(os.environ.get("SENTINEL_PORT", "8000"))
    run_server(host="0.0.0.0", port=port)
