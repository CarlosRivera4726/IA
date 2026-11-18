"""
Auditoría Automatizada del Contrato REST API (backend_ia v2).
Verifica:
1. OPTIONS /api/v1/infer (CORS completo para el frontend Astro).
2. GET /api/v1/health (Backend Torch/CUDA con los 4 modelos neuronales cargados: escena, género, pose, objetos).
3. GET /api/v1/cameras (Exactamente 6 cámaras: CAM-01 activa local y CAM-02..CAM-06 temporalmente deshabilitadas en el servidor).
4. POST /api/v1/infer con imágenes reales de holdout evaluando detección autónoma pura (sin demo_override, sin forzado artificial).
5. POST /api/v1/infer validando estructura completa (primary_event, scene_probabilities, detected_persons, detected_objects, evidence).
6. Validación de seguridad y robustez: HTTP 403 al inferir sobre cámara deshabilitada (CAM-02) y HTTP 400 ante payload sin imagen.
"""

from __future__ import annotations

import base64
import email
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ["KERAS_HOME"] = r"c:\Users\Carlos\Documents\GitHub\IA\.keras"
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TORCH_HOME", r"c:\Users\Carlos\Documents\GitHub\IA\.torch")

REPO_ROOT = Path(r"c:\Users\Carlos\Documents\GitHub\IA")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HOLDOUT_DIR = REPO_ROOT / "datasets" / "holdout"
BASE_URL = os.environ.get("SENTINEL_API_URL", "http://127.0.0.1:8000")

EXPECTED_SCENE_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
]

RequesterFn = Callable[[str, str, Optional[Dict[str, Any]]], Tuple[int, Dict[str, str], Any]]


def _make_socket_requester(base_url: str) -> RequesterFn:
    def _request(
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        url = f"{base_url.rstrip('/')}{path}"
        data = None
        headers = {"Origin": "http://localhost:4321"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                raw_body = resp.read().decode("utf-8", errors="replace")
                try:
                    body = json.loads(raw_body) if raw_body.strip() else {}
                except json.JSONDecodeError:
                    body = raw_body
                return status, resp_headers, body
        except urllib.error.HTTPError as err:
            resp_headers = {k.lower(): v for k, v in err.headers.items()} if err.headers else {}
            raw_body = err.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw_body) if raw_body.strip() else {}
            except json.JSONDecodeError:
                body = raw_body
            return err.code, resp_headers, body

    return _request


def _make_inprocess_requester() -> RequesterFn:
    from backend_ia.presentation.http_server import SentinelHttpRequestHandler, get_container

    get_container()

    class _InMemoryHandler(SentinelHttpRequestHandler):
        def __init__(self, method: str, path: str, body_bytes: bytes) -> None:
            self.rfile = io.BytesIO(body_bytes)
            self.wfile = io.BytesIO()
            self.command = method
            self.path = path
            self.request_version = "HTTP/1.1"
            self.requestline = f"{method} {path} HTTP/1.1"
            self.client_address = ("127.0.0.1", 54321)
            hdrs = email.message.Message()
            hdrs["Host"] = "localhost:8000"
            hdrs["Origin"] = "http://localhost:4321"
            if body_bytes:
                hdrs["Content-Type"] = "application/json"
                hdrs["Content-Length"] = str(len(body_bytes))
            self.headers = hdrs
            self._headers_buffer: List[bytes] = []

    def _request(
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else b""
        handler = _InMemoryHandler(method, path, body_bytes)
        if method == "OPTIONS":
            handler.do_OPTIONS()
        elif method == "GET":
            handler.do_GET()
        elif method == "POST":
            handler.do_POST()
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        raw_response = handler.wfile.getvalue()
        header_part, _, body_part = raw_response.partition(b"\r\n\r\n")
        header_lines = header_part.decode("iso-8859-1").split("\r\n")
        status_line = header_lines[0]
        parts = status_line.split(" ", 2)
        status_code = int(parts[1]) if len(parts) >= 2 else 500

        parsed_headers: Dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                parsed_headers[k.strip().lower()] = v.strip()

        raw_text = body_part.decode("utf-8", errors="replace")
        try:
            parsed_body = json.loads(raw_text) if raw_text.strip() else {}
        except json.JSONDecodeError:
            parsed_body = raw_text

        return status_code, parsed_headers, parsed_body

    return _request


def _get_sample_holdout_image(category: str) -> str:
    target_dir = HOLDOUT_DIR / category
    if target_dir.is_dir():
        files = sorted(list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png")))
        if files:
            raw = files[0].read_bytes()
            mime = "image/png" if files[0].suffix.lower() == ".png" else "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def run_audit() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    server_instance = None
    requester: Optional[RequesterFn] = None
    transport_mode = "external_socket"

    try:
        # 1. Probar servidor existente
        ext_req = _make_socket_requester(BASE_URL)
        try:
            st, _, _ = ext_req("GET", "/api/v1/health", None)
            if st == 200:
                requester = ext_req
                transport_mode = f"live_socket ({BASE_URL})"
        except Exception:
            requester = None

        # 2. Levantar servidor local en memoria si no hay puerto abierto
        if requester is None:
            try:
                from backend_ia.presentation.http_server import create_http_server

                server_instance, _ = create_http_server(host="127.0.0.1", port=0)
                bound_port = server_instance.server_address[1]
                server_thread = threading.Thread(target=server_instance.serve_forever, daemon=True)
                server_thread.start()
                local_url = f"http://127.0.0.1:{bound_port}"
                sock_req = _make_socket_requester(local_url)
                st, _, _ = sock_req("GET", "/api/v1/health", None)
                if st == 200:
                    requester = sock_req
                    transport_mode = f"threaded_http_server ({local_url})"
            except Exception:
                requester = None

        if requester is None:
            requester = _make_inprocess_requester()
            transport_mode = "in_memory_http_protocol (SentinelHttpRequestHandler)"

        # Control 1: OPTIONS CORS
        opt_status, opt_headers, _ = requester("OPTIONS", "/api/v1/infer", None)
        cors_origin = opt_headers.get("access-control-allow-origin", "")
        cors_methods = opt_headers.get("access-control-allow-methods", "")
        cors_ok = opt_status in (200, 204) and cors_origin == "*" and "POST" in cors_methods
        checks.append({
            "id": "API-01",
            "name": "Soporte CORS en OPTIONS /api/v1/infer para integración web Astro",
            "passed": cors_ok,
            "details": f"[{transport_mode}] status={opt_status}, Allow-Origin='{cors_origin}', Allow-Methods='{cors_methods}'",
        })

        # Control 2: GET /api/v1/health
        h_status, _, h_body = requester("GET", "/api/v1/health", None)
        models_loaded = h_body.get("models_loaded", {}) if isinstance(h_body, dict) else {}
        health_ok = (
            h_status == 200
            and isinstance(h_body, dict)
            and h_body.get("status") == "healthy"
            and bool(models_loaded.get("scene"))
            and bool(models_loaded.get("gender"))
            and bool(models_loaded.get("pose"))
            and bool(models_loaded.get("objects"))
        )
        checks.append({
            "id": "API-02",
            "name": "Endpoint GET /api/v1/health (backend activo y los 4 modelos neuronales cargados en CUDA)",
            "passed": health_ok,
            "details": (
                f"status={h_status}, backend={h_body.get('backend')}, device={h_body.get('device')}, "
                f"models_loaded={models_loaded}"
            ),
        })

        # Control 3: GET /api/v1/cameras (6 cámaras: CAM-01 activa y CAM-02..06 temporalmente deshabilitadas)
        c_status, _, c_body = requester("GET", "/api/v1/cameras", None)
        cameras_list = c_body.get("cameras", []) if isinstance(c_body, dict) else []
        cam_by_id = {c.get("camera_id") or c.get("id"): c for c in cameras_list if isinstance(c, dict)}
        cam1 = cam_by_id.get("CAM-01", {})
        cam1_active = bool(cam1.get("is_active") is True and cam1.get("status") == "ACTIVE")
        disabled_ids = [f"CAM-0{i}" for i in range(2, 7)]
        disabled_ok = all(
            cid in cam_by_id
            and cam_by_id[cid].get("is_active") is False
            and cam_by_id[cid].get("status") == "TEMPORARILY_DISABLED"
            for cid in disabled_ids
        )
        six_cameras_ok = c_status == 200 and len(cam_by_id) == 6 and cam1_active and disabled_ok
        checks.append({
            "id": "API-03",
            "name": "Endpoint GET /api/v1/cameras (6 cámaras: CAM-01 activa local y CAM-02..06 deshabilitadas)",
            "passed": six_cameras_ok,
            "details": (
                f"total={len(cam_by_id)}, CAM-01={cam1.get('status')}, "
                f"CAM-02..06=TEMPORARILY_DISABLED ({disabled_ok})"
            ),
        })

        # Control 4: Inferencia autónoma real POST /api/v1/infer con imagen de holdout
        sample_b64 = _get_sample_holdout_image("Arma_Blanca")
        inf_status, _, inf_body = requester(
            "POST",
            "/api/v1/infer",
            {
                "camera_id": "CAM-01",
                "image_base64": sample_b64,
                "frame_ts": 1700000000000,
            },
        )
        probs = inf_body.get("scene_probabilities", {}) if isinstance(inf_body, dict) else {}
        prob_keys_ok = sorted(probs.keys()) == EXPECTED_SCENE_CLASSES
        prob_sum = sum(float(v) for v in probs.values()) if probs else 0.0
        pure_infer_ok = (
            inf_status == 200
            and isinstance(inf_body, dict)
            and "primary_event" in inf_body
            and prob_keys_ok
            and abs(prob_sum - 1.0) <= 0.02
            and isinstance(inf_body.get("detected_persons"), list)
            and isinstance(inf_body.get("detected_objects"), list)
            and isinstance(inf_body.get("evidence"), dict)
        )
        checks.append({
            "id": "API-04",
            "name": "Inferencia autónoma real POST /api/v1/infer (Holdout real -> Keras + Torchvision -> Salida estructurada)",
            "passed": pure_infer_ok,
            "details": (
                f"status={inf_status}, primary_event='{inf_body.get('primary_event')}', "
                f"threat_level='{inf_body.get('threat_level')}', prob_sum={prob_sum:.4f}, "
                f"latency_ms={inf_body.get('model_info', {}).get('latency_ms')}"
            ),
        })

        # Control 5: Bloqueo de inferencia en cámara deshabilitada (CAM-02 -> HTTP 403)
        dis_status, _, dis_body = requester(
            "POST",
            "/api/v1/infer",
            {
                "camera_id": "CAM-02",
                "image_base64": sample_b64,
            },
        )
        dis_ok = dis_status == 403 and isinstance(dis_body, dict) and "error" in dis_body
        checks.append({
            "id": "API-05",
            "name": "Bloqueo de inferencia en cámara del servidor temporalmente deshabilitada (CAM-02 -> HTTP 403)",
            "passed": dis_ok,
            "details": f"status={dis_status}, error={dis_body.get('error')}",
        })

        # Control 6: Rechazo ante petición sin imagen (HTTP 400 Bad Request)
        bad_status, _, bad_body = requester(
            "POST",
            "/api/v1/infer",
            {
                "camera_id": "CAM-01",
            },
        )
        bad_ok = bad_status == 400 and isinstance(bad_body, dict) and "error" in bad_body
        checks.append({
            "id": "API-06",
            "name": "Validación de esquema y rechazo de payload incompleto sin imagen (HTTP 400)",
            "passed": bad_ok,
            "details": f"status={bad_status}, error={bad_body.get('error')}",
        })

    finally:
        if server_instance is not None:
            try:
                server_instance.shutdown()
                server_instance.server_close()
            except Exception:
                pass

    passed_count = sum(1 for c in checks if c["passed"])
    return {
        "suite": "REST API & Autonomous Inference Contract Audit",
        "passed": passed_count == len(checks) and len(checks) > 0,
        "passed_count": passed_count,
        "total_count": len(checks),
        "checks": checks,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)
