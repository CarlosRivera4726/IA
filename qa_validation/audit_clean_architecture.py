"""
Auditoría Automatizada de Arquitectura Limpia (Clean Architecture).
Verifica mediante AST (Python) y análisis léxico/sintáctico (TypeScript/Astro)
que las capas de Dominio y Aplicación en `backend_ia/` y `web_security/`
respeten estrictamente la Regla de Dependencia.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(r"c:\Users\Carlos\Documents\GitHub\IA")
BACKEND_DIR = REPO_ROOT / "backend_ia"
WEB_SRC_DIR = REPO_ROOT / "web_security" / "src"

FORBIDDEN_BACKEND_IMPORTS = {
    "keras",
    "tensorflow",
    "torch",
    "torchvision",
    "numpy",
    "np",
    "PIL",
    "pil",
    "cv2",
    "http",
    "fastapi",
    "uvicorn",
    "starlette",
    " infrastructure",
    "infrastructure",
    "presentation",
}

FORBIDDEN_FRONTEND_PATTERNS = [
    (r"\bwindow\.", "window."),
    (r"\bdocument\.", "document."),
    (r"\bnavigator\.", "navigator."),
    (r"\bfetch\s*\(", "fetch("),
    (r"\blocalStorage\b", "localStorage"),
]


def _extract_python_imports(py_file: Path) -> List[str]:
    """Extrae todos los nombres de módulos importados usando el árbol AST."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return imports


def _has_formal_ports(py_files: List[Path]) -> bool:
    """Verifica si los archivos definen interfaces formales con Protocol o ABC."""
    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name in ("Protocol", "ABC"):
                        return True
    return False


def _strip_ts_comments(source: str) -> str:
    """Elimina comentarios de bloque y de línea en TypeScript/JS para evitar falsos positivos."""
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", source)
    no_line = re.sub(r"//.*", "", no_block)
    return no_line


def audit_backend_clean_architecture() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    domain_dir = BACKEND_DIR / "domain"
    app_dir = BACKEND_DIR / "application"
    infra_dir = BACKEND_DIR / "infrastructure"
    pres_dir = BACKEND_DIR / "presentation"

    layers_exist = all(d.is_dir() for d in (domain_dir, app_dir, infra_dir, pres_dir))
    checks.append({
        "id": "CA-BACK-01",
        "name": "Existencia de las 4 capas en backend_ia/ (domain, application, infrastructure, presentation)",
        "passed": layers_exist,
        "details": (
            "Las 4 carpetas de capas existen en backend_ia/."
            if layers_exist
            else f"Capas faltantes en {BACKEND_DIR}: "
                 f"{[d.name for d in (domain_dir, app_dir, infra_dir, pres_dir) if not d.is_dir()]}"
        ),
    })

    # Auditar domain/
    domain_files = sorted(domain_dir.rglob("*.py")) if domain_dir.is_dir() else []
    non_init_domain = [f for f in domain_files if f.name != "__init__.py" or f.stat().st_size > 0]
    domain_violations: List[str] = []

    for py_file in domain_files:
        try:
            imports = _extract_python_imports(py_file)
            for imp in imports:
                root_mod = imp.split(".")[0]
                parts = imp.split(".")
                if root_mod in FORBIDDEN_BACKEND_IMPORTS or any(
                    p in FORBIDDEN_BACKEND_IMPORTS for p in parts
                ):
                    domain_violations.append(f"{py_file.name} importa '{imp}'")
        except SyntaxError as exc:
            domain_violations.append(f"Error de sintaxis en {py_file.name}: {exc}")

    checks.append({
        "id": "CA-BACK-02",
        "name": "Aislamiento puro de backend_ia/domain/ via AST (sin Keras, TF, Torch, NumPy, PIL, CV2, HTTP, FastAPI)",
        "passed": bool(non_init_domain) and len(domain_violations) == 0,
        "details": (
            f"Analizados {len(domain_files)} archivos en domain/ con 0 violaciones de dependencia."
            if (non_init_domain and not domain_violations)
            else f"Violaciones o archivos faltantes: {domain_violations or 'No hay módulos en domain/'}"
        ),
    })

    # Auditar application/
    app_files = sorted(app_dir.rglob("*.py")) if app_dir.is_dir() else []
    non_init_app = [f for f in app_files if f.name != "__init__.py" or f.stat().st_size > 0]
    app_violations: List[str] = []

    for py_file in app_files:
        try:
            imports = _extract_python_imports(py_file)
            for imp in imports:
                root_mod = imp.split(".")[0]
                parts = imp.split(".")
                if root_mod in FORBIDDEN_BACKEND_IMPORTS or any(
                    p in FORBIDDEN_BACKEND_IMPORTS for p in parts
                ):
                    app_violations.append(f"{py_file.name} importa '{imp}'")
        except SyntaxError as exc:
            app_violations.append(f"Error de sintaxis en {py_file.name}: {exc}")

    has_ports = _has_formal_ports(app_files + domain_files) if (app_files or domain_files) else False

    checks.append({
        "id": "CA-BACK-03",
        "name": "Regla de Dependencia y Puertos formales (Protocol/ABC) en backend_ia/application/",
        "passed": bool(non_init_app) and len(app_violations) == 0 and has_ports,
        "details": (
            f"Analizados {len(app_files)} archivos en application/; puertos formales detectados={has_ports}, violaciones=0."
            if (non_init_app and not app_violations and has_ports)
            else f"Estado: archivos={len(non_init_app)}, has_ports={has_ports}, violaciones={app_violations}"
        ),
    })

    return checks


def audit_frontend_clean_architecture() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    domain_dir = WEB_SRC_DIR / "domain"
    app_dir = WEB_SRC_DIR / "application"
    infra_dir = WEB_SRC_DIR / "infrastructure"
    pres_dir = WEB_SRC_DIR / "presentation"

    layers_exist = all(d.is_dir() for d in (domain_dir, app_dir, infra_dir, pres_dir))
    checks.append({
        "id": "CA-FRONT-01",
        "name": "Existencia de las 4 capas en web_security/src/ (domain, application, infrastructure, presentation)",
        "passed": layers_exist,
        "details": (
            "Las 4 carpetas de capas existen en web_security/src/."
            if layers_exist
            else f"Capas faltantes en {WEB_SRC_DIR}: "
                 f"{[d.name for d in (domain_dir, app_dir, infra_dir, pres_dir) if not d.is_dir()]}"
        ),
    })

    for layer_id, layer_name, layer_dir in [
        ("CA-FRONT-02", "web_security/src/domain/", domain_dir),
        ("CA-FRONT-03", "web_security/src/application/", app_dir),
    ]:
        files = (
            sorted(
                [
                    f
                    for f in layer_dir.rglob("*")
                    if f.is_file() and f.suffix in (".ts", ".js", ".mjs", ".astro")
                ]
            )
            if layer_dir.is_dir()
            else []
        )
        violations: List[str] = []
        for f in files:
            raw = f.read_text(encoding="utf-8")
            clean = _strip_ts_comments(raw)
            for pattern, label in FORBIDDEN_FRONTEND_PATTERNS:
                if re.search(pattern, clean):
                    violations.append(f"{f.name} usa '{label}'")
            if re.search(r"from\s+['\"].*(infrastructure|presentation)", clean):
                violations.append(f"{f.name} importa capa externa (infrastructure/presentation)")

        checks.append({
            "id": layer_id,
            "name": f"Aislamiento de {layer_name} (sin window., document., navigator., fetch(, localStorage)",
            "passed": len(files) > 0 and len(violations) == 0,
            "details": (
                f"Analizados {len(files)} archivos en {layer_name} con 0 violaciones."
                if (files and not violations)
                else f"Archivos={len(files)}, violaciones={violations}"
            ),
        })

    return checks


def run_audit() -> Dict[str, Any]:
    checks = audit_backend_clean_architecture() + audit_frontend_clean_architecture()
    passed_count = sum(1 for c in checks if c["passed"])
    return {
        "suite": "Clean Architecture Audit (AST & Layer Isolation)",
        "passed": passed_count == len(checks),
        "passed_count": passed_count,
        "total_count": len(checks),
        "checks": checks,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)

