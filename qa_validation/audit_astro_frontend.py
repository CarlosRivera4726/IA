"""
Auditoría Automatizada del Frontend Web en Astro (`web_security/`) y Directrices `frontend-design`.
Verifica:
1. Estructura del proyecto Astro (`package.json`, `astro.config.*`, `src/pages/index.astro`)
   y sus 4 capas (`domain`, `application`, `infrastructure`, `presentation`).
2. Presencia de Landing Page y Grilla de 6 Cámaras (`CAM-01` conectada al dispositivo local
   mediante `getUserMedia` + fallback canvas, y `CAM-02`..`CAM-06` temporalmente deshabilitadas).
3. Integración con `/api/v1/infer` y soporte de las 7 clases canónicas de seguridad.
4. Cumplimiento de las directrices de la skill `frontend-design` (paleta de colores calibrada
   `#111417`, `#191E24`, `#ECEAE4`, `#E03E2D`, `#D99B26`, `#289E68`, y tipografías
   `Space Grotesk` e `IBM Plex Mono`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(r"c:\Users\Carlos\Documents\GitHub\IA")
WEB_DIR = REPO_ROOT / "web_security"
SRC_DIR = WEB_DIR / "src"

EXPECTED_HEX_TOKENS = [
    "#111417",
    "#191E24",
    "#ECEAE4",
    "#E03E2D",
    "#D99B26",
    "#289E68",
]

CANONICAL_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
    "Senal_Auxilio_Hombre",
    "Senal_Auxilio_Mujer",
]


def _collect_all_source_text() -> str:
    if not WEB_DIR.is_dir():
        return ""
    chunks: List[str] = []
    for ext in ("*.astro", "*.ts", "*.js", "*.mjs", "*.css", "*.json"):
        for f in sorted(WEB_DIR.rglob(ext)):
            if "node_modules" in f.parts or ".astro" in f.parts or "dist" in f.parts:
                continue
            try:
                chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return "\n".join(chunks)


def run_audit() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    pkg_json = WEB_DIR / "package.json"
    index_astro = SRC_DIR / "pages" / "index.astro"
    layers = [SRC_DIR / name for name in ("domain", "application", "infrastructure", "presentation")]

    # 1. Estructura Astro y Clean Architecture
    has_astro_dep = False
    if pkg_json.is_file():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            has_astro_dep = "astro" in deps
        except Exception:
            has_astro_dep = False

    structure_ok = pkg_json.is_file() and has_astro_dep and index_astro.is_file() and all(d.is_dir() for d in layers)
    checks.append({
        "id": "WEB-01",
        "name": "Estructura Astro (`package.json`, `src/pages/index.astro`) y 4 capas Clean Architecture",
        "passed": structure_ok,
        "details": (
            f"package.json={pkg_json.is_file()}, astro_dep={has_astro_dep}, "
            f"index.astro={index_astro.is_file()}, capas={[d.name for d in layers if d.is_dir()]}"
        ),
    })

    all_text = _collect_all_source_text()
    all_text_upper = all_text.upper()

    # 2. Grilla de 6 cámaras (CAM-01 activa con getUserMedia + CAM-02..CAM-06 deshabilitadas)
    cam_ids_present = [f"CAM-0{i}" in all_text for i in range(1, 7)]
    has_get_user_media = "getUserMedia" in all_text
    has_disabled_copy = any(
        kw in all_text.lower()
        for kw in ("temporalmente deshabilitada", "deshabilitada", "temporarily_disabled", "en espera")
    )
    cameras_ok = all(cam_ids_present) and has_get_user_media and has_disabled_copy
    checks.append({
        "id": "WEB-02",
        "name": "Grilla de 6 cámaras: CAM-01 activa (`getUserMedia`) y CAM-02..CAM-06 temporalmente deshabilitadas",
        "passed": cameras_ok,
        "details": (
            f"CAM-01..06 presentes={all(cam_ids_present)}, getUserMedia={has_get_user_media}, "
            f"estado deshabilitado en servidor={has_disabled_copy}"
        ),
    })

    # 3. Consumo de `/api/v1/infer` y soporte de las 7 clases de seguridad
    has_infer_endpoint = "/api/v1/infer" in all_text
    classes_present = [cls for cls in CANONICAL_CLASSES if cls in all_text]
    integration_ok = has_infer_endpoint and len(classes_present) == len(CANONICAL_CLASSES)
    checks.append({
        "id": "WEB-03",
        "name": "Integración con `/api/v1/infer` y soporte de las 7 clases canónicas de detección",
        "passed": integration_ok,
        "details": (
            f"endpoint `/api/v1/infer`={has_infer_endpoint}, "
            f"clases soportadas={len(classes_present)}/7 ({classes_present})"
        ),
    })

    # 4. Directrices de `frontend-design` (Paleta óptica y tipografías Space Grotesk + IBM Plex Mono)
    matched_colors = [hx for hx in EXPECTED_HEX_TOKENS if hx.upper() in all_text_upper]
    has_space_grotesk = "space grotesk" in all_text.lower()
    has_ibm_plex_mono = "ibm plex mono" in all_text.lower()
    design_ok = len(matched_colors) >= 5 and has_space_grotesk and has_ibm_plex_mono
    checks.append({
        "id": "WEB-04",
        "name": "Cumplimiento de `frontend-design` (Paleta óptica calibrada + Space Grotesk & IBM Plex Mono)",
        "passed": design_ok,
        "details": (
            f"Colores encontrados={matched_colors} ({len(matched_colors)}/6), "
            f"Space Grotesk={has_space_grotesk}, IBM Plex Mono={has_ibm_plex_mono}"
        ),
    })

    passed_count = sum(1 for c in checks if c["passed"])
    return {
        "suite": "Astro Web & frontend-design Skill Audit",
        "passed": passed_count == len(checks),
        "passed_count": passed_count,
        "total_count": len(checks),
        "checks": checks,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)

