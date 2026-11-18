"""Punto de entrada cuando se ejecuta python -m backend_ia."""

from backend_ia.main import run_server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("SENTINEL_PORT", "8000"))
    run_server(host="0.0.0.0", port=port)

