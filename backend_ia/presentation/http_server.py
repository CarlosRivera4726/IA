"""
Servidor HTTP de presentación para el servicio SentinelCV (Clean Architecture).
Implementa endpoints REST con soporte completo de CORS y formato de contrato v2:
- GET  /api/v1/health
- GET  /api/v1/cameras
- POST /api/v1/infer
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from backend_ia.application.use_cases import (
    AnalyzeCameraFrameUseCase,
    CameraDisabledError,
    GetCameraGridStatusUseCase,
    InvalidImagePayloadError,
)
from backend_ia.infrastructure.camera_repository import InMemoryCameraRepository
from backend_ia.infrastructure.keras_model_adapter import KerasSecurityModelAdapter

# Instancias compartidas de infraestructura y aplicación
_camera_repo = InMemoryCameraRepository()
_model_adapter = KerasSecurityModelAdapter()
_analyze_use_case = AnalyzeCameraFrameUseCase(_model_adapter, _camera_repo)
_get_grid_use_case = GetCameraGridStatusUseCase(_camera_repo)


def get_container():
    return {
        "camera_repo": _camera_repo,
        "model_adapter": _model_adapter,
        "analyze_use_case": _analyze_use_case,
        "get_grid_use_case": _get_grid_use_case,
    }


class SentinelHttpRequestHandler(BaseHTTPRequestHandler):
    """Manejador de peticiones HTTP para SentinelCV con soporte CORS total."""

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"
        )

    def _send_json_response(self, status_code: int, data: object) -> None:
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(payload)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # El cliente aborto la conexion (timeout o recarga de pagina en navegador)
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path == "/api/v1/health":
            health_data = _model_adapter.get_model_health()
            self._send_json_response(200, health_data)
        elif path == "/api/v1/cameras":
            cameras = _get_grid_use_case.execute()
            cam_dicts = [c.to_dict() for c in cameras]
            self._send_json_response(200, {
                "cameras": cam_dicts,
                "total_cameras": len(cam_dicts),
                "active_cameras": sum(1 for c in cameras if c.is_active),
                "disabled_cameras": sum(1 for c in cameras if not c.is_active),
            })
        else:
            self._send_json_response(
                404, {"error": "Not Found", "message": f"Endpoint '{self.path}' no encontrado."}
            )

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path != "/api/v1/infer":
            self._send_json_response(
                404, {"error": "Not Found", "message": f"Endpoint '{self.path}' no encontrado."}
            )
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json_response(
                400, {"error": "Bad Request", "message": "Cuerpo de la petición vacío."}
            )
            return

        try:
            body = self.rfile.read(content_length)
            req_data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json_response(
                400, {"error": "Bad Request", "message": f"JSON inválido: {e}"}
            )
            return

        camera_id = str(req_data.get("camera_id", "CAM-01"))
        image_base64 = req_data.get("image_base64", "")
        frame_ts: Optional[int] = req_data.get("frame_ts")

        try:
            result = _analyze_use_case.execute(
                camera_id=camera_id,
                image_base64=image_base64,
                frame_ts=frame_ts,
            )
            self._send_json_response(200, result.to_dict())
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass
        except InvalidImagePayloadError as e:
            self._send_json_response(400, {"error": "Bad Request", "message": str(e)})
        except CameraDisabledError as e:
            self._send_json_response(403, {"error": "Forbidden", "message": str(e)})
        except Exception as e:
            self._send_json_response(500, {"error": "Internal Error", "message": str(e)})

    def log_message(self, format: str, *args) -> None:
        pass


def create_http_server(host: str = "127.0.0.1", port: int = 8000) -> Tuple[ThreadingHTTPServer, int]:
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SentinelHttpRequestHandler)
    bound_port = httpd.server_address[1]
    return httpd, bound_port


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    httpd, bound_port = create_http_server(host, port)
    print(f"Servidor SentinelCV v2 activo en http://{host}:{bound_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("SENTINEL_PORT", "8000"))
    run_server(host="0.0.0.0", port=port)
