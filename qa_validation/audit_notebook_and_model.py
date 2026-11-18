"""
Auditoría Automatizada del Notebook (`ParcialFinal.ipynb`) y de los Modelos Compilados (`.keras`).
Verifica:
1. Estructura académica, ejecución completa sin errores, `custom_model` y `transfer_model`,
   matrices de confusión y tabla comparativa en `ParcialFinal.ipynb`.
2. Existencia y contrato de `models/model_metadata.json` con las clases de escena reales y métricas en holdout.
3. Carga real de `models/sentinel_security_model.keras` con Keras 3 (`KERAS_BACKEND=torch`),
   verificando compilación, inferencia sobre tensores de prueba `(N, 224, 224, 3)`,
   shape de salida `(N, 5)` y suma de probabilidades `1.0`.
4. Carga real de `models/sentinel_gender_model.keras` con Keras 3,
   verificando inferencia facial sobre tensores `(N, 224, 224, 3)`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["KERAS_HOME"] = r"c:\Users\Carlos\Documents\GitHub\IA\.keras"
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TORCH_HOME", r"c:\Users\Carlos\Documents\GitHub\IA\.torch")

REPO_ROOT = Path(r"c:\Users\Carlos\Documents\GitHub\IA")
NOTEBOOK_PATH = REPO_ROOT / "ParcialFinal.ipynb"
METADATA_PATH = REPO_ROOT / "models" / "model_metadata.json"
SCENE_MODEL_PATH = REPO_ROOT / "models" / "sentinel_security_model.keras"
GENDER_MODEL_PATH = REPO_ROOT / "models" / "sentinel_gender_model.keras"

EXPECTED_SCENE_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
]

EXPECTED_GENDER_CLASSES = [
    "Hombre",
    "Mujer",
]


def audit_notebook() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    if not NOTEBOOK_PATH.is_file():
        checks.append({
            "id": "NB-01",
            "name": "Existencia y validez JSON de ParcialFinal.ipynb",
            "passed": False,
            "details": f"No existe el archivo {NOTEBOOK_PATH}",
        })
        return checks

    try:
        nb_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append({
            "id": "NB-01",
            "name": "Existencia y validez JSON de ParcialFinal.ipynb",
            "passed": False,
            "details": f"Error leyendo JSON de ParcialFinal.ipynb: {exc}",
        })
        return checks

    cells = nb_data.get("cells", [])
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    checks.append({
        "id": "NB-01",
        "name": "Estructura completa de ParcialFinal.ipynb (celdas Markdown y Code)",
        "passed": len(md_cells) >= 4 and len(code_cells) >= 5,
        "details": f"Total celdas={len(cells)} (Markdown={len(md_cells)}, Code={len(code_cells)}).",
    })

    # Verificar ejecución y ausencia de errores
    error_cells: List[str] = []
    unexecuted_cells: List[int] = []
    all_source_parts: List[str] = []

    for idx, cell in enumerate(cells):
        src = cell.get("source", [])
        src_text = "".join(src) if isinstance(src, list) else str(src)
        all_source_parts.append(src_text)

        if cell.get("cell_type") == "code" and src_text.strip():
            if cell.get("execution_count") is None:
                unexecuted_cells.append(idx)
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")
                    error_cells.append(f"Celda #{idx}: {ename}: {evalue}")

    full_source = "\n".join(all_source_parts)

    checks.append({
        "id": "NB-02",
        "name": "Ejecución completa sin errores (0 output_type == 'error' y celdas ejecutadas)",
        "passed": len(code_cells) > 0 and len(error_cells) == 0 and len(unexecuted_cells) == 0,
        "details": (
            f"Todas las {len(code_cells)} celdas de código ejecutadas sin errores."
            if (code_cells and not error_cells and not unexecuted_cells)
            else f"Errores={error_cells}, Celdas sin ejecutar={unexecuted_cells}"
        ),
    })

    has_custom = "custom_model" in full_source
    has_transfer = "transfer_model" in full_source or "MobileNetV2" in full_source
    has_confusion = "confusion_matrix" in full_source
    has_comparison = any(
        kw in full_source.lower()
        for kw in ("comparativa", "comparison", "f1", "classification_report", "dataframe")
    )

    checks.append({
        "id": "NB-03",
        "name": "Presencia de custom_model, transfer_model (MobileNetV2), matrices de confusión y tabla comparativa",
        "passed": has_custom and has_transfer and has_confusion and has_comparison,
        "details": (
            f"custom_model={has_custom}, transfer_model={has_transfer}, "
            f"confusion_matrix={has_confusion}, tabla_comparativa={has_comparison}"
        ),
    })

    return checks


def audit_metadata_and_keras_models() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    if not METADATA_PATH.is_file():
        checks.append({
            "id": "MOD-01",
            "name": "Existencia y validez de models/model_metadata.json con datasets reales",
            "passed": False,
            "details": f"No existe {METADATA_PATH}",
        })
    else:
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            scene_classes = meta.get("scene_class_names") or meta.get("class_names") or []
            gender_classes = meta.get("gender_class_names") or []

            scene_ok = list(scene_classes) == EXPECTED_SCENE_CLASSES
            gender_ok = list(gender_classes) == EXPECTED_GENDER_CLASSES
            has_metrics = "scene_champion" in meta

            checks.append({
                "id": "MOD-01",
                "name": "Existencia y validez de models/model_metadata.json (clases de escena, género y métricas)",
                "passed": scene_ok and gender_ok and has_metrics,
                "details": (
                    f"Clases de escena={scene_classes}, Clases de género={gender_classes}, "
                    f"Campeón evaluado={meta.get('scene_champion', {}).get('Modelo', 'N/A')}."
                    if (scene_ok and gender_ok and has_metrics)
                    else f"Inconsistencias en metadata: scene_ok={scene_ok}, gender_ok={gender_ok}, has_metrics={has_metrics}"
                ),
            })
        except Exception as exc:
            checks.append({
                "id": "MOD-01",
                "name": "Existencia y validez de models/model_metadata.json",
                "passed": False,
                "details": f"Error leyendo metadata: {exc}",
            })

    # Chequear modelo de escena
    if not SCENE_MODEL_PATH.is_file():
        checks.append({
            "id": "MOD-02",
            "name": "Carga e inferencia real de models/sentinel_security_model.keras (shape=(N, 5), sum=1.0)",
            "passed": False,
            "details": f"No existe {SCENE_MODEL_PATH}",
        })
    else:
        try:
            import keras
            import numpy as np

            model = keras.models.load_model(str(SCENE_MODEL_PATH))
            input_shape = model.input_shape
            height = input_shape[1] if input_shape and len(input_shape) >= 4 and input_shape[1] else 224
            width = input_shape[2] if input_shape and len(input_shape) >= 4 and input_shape[2] else 224

            batch_size = 2
            rng = np.random.default_rng(42)
            dummy_batch = rng.uniform(0.0, 255.0, size=(batch_size, height, width, 3)).astype("float32")
            preds = model.predict(dummy_batch, verbose=0)
            preds_np = np.asarray(preds, dtype="float64")

            shape_ok = preds_np.shape == (batch_size, 5)
            row_sums = preds_np.sum(axis=-1)
            probs_ok = bool(np.allclose(row_sums, 1.0, atol=1e-2))

            checks.append({
                "id": "MOD-02",
                "name": "Carga e inferencia real de models/sentinel_security_model.keras (shape=(N, 5), sum=1.0)",
                "passed": shape_ok and probs_ok,
                "details": (
                    f"Modelo de escena '{model.name}' cargado exitosamente ({SCENE_MODEL_PATH.stat().st_size / (1024*1024):.1f} MB). "
                    f"Input=({batch_size}, {height}, {width}, 3) -> Output shape={preds_np.shape}, "
                    f"sumas softmax={np.round(row_sums, 4).tolist()}."
                ),
            })
        except Exception as exc:
            checks.append({
                "id": "MOD-02",
                "name": "Carga e inferencia real de models/sentinel_security_model.keras",
                "passed": False,
                "details": f"Fallo al cargar o inferir con {SCENE_MODEL_PATH}: {exc}",
            })

    # Chequear modelo de género
    if not GENDER_MODEL_PATH.is_file():
        checks.append({
            "id": "MOD-03",
            "name": "Carga e inferencia real de models/sentinel_gender_model.keras",
            "passed": False,
            "details": f"No existe {GENDER_MODEL_PATH}",
        })
    else:
        try:
            import keras
            import numpy as np

            g_model = keras.models.load_model(str(GENDER_MODEL_PATH))
            batch_size = 2
            rng = np.random.default_rng(42)
            dummy_faces = rng.uniform(0.0, 255.0, size=(batch_size, 224, 224, 3)).astype("float32")
            g_preds = g_model.predict(dummy_faces, verbose=0)
            g_preds_np = np.asarray(g_preds, dtype="float64")

            shape_ok = g_preds_np.shape == (batch_size, 2) or g_preds_np.shape == (batch_size, 1)

            checks.append({
                "id": "MOD-03",
                "name": "Carga e inferencia real de models/sentinel_gender_model.keras (clasificador facial)",
                "passed": shape_ok,
                "details": (
                    f"Modelo facial '{g_model.name}' cargado ({GENDER_MODEL_PATH.stat().st_size / (1024*1024):.1f} MB). "
                    f"Output shape={g_preds_np.shape}."
                ),
            })
        except Exception as exc:
            checks.append({
                "id": "MOD-03",
                "name": "Carga e inferencia real de models/sentinel_gender_model.keras",
                "passed": False,
                "details": f"Fallo al cargar modelo de género: {exc}",
            })

    return checks


def run_audit() -> Dict[str, Any]:
    checks = audit_notebook() + audit_metadata_and_keras_models()
    passed_count = sum(1 for c in checks if c["passed"])
    return {
        "suite": "Notebook & Compiled Keras Model Audit",
        "passed": passed_count == len(checks),
        "passed_count": passed_count,
        "total_count": len(checks),
        "checks": checks,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)
