"""Script de inicio rapido para el servidor de inferencia SentinelCV."""

from backend_ia.main import run_server

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000)

