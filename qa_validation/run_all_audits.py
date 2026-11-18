"""
Orquestador Maestro de Auditoría QA (`qa_validation/run_all_audits.py`).
Ejecuta las 4 suites de validación automatizada sin modificar ningún archivo
fuera de `qa_validation/` y genera `qa_validation/QA_AUDIT_REPORT.md`.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

QA_DIR = Path(__file__).resolve().parent
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))

import audit_api_contract
import audit_astro_frontend
import audit_clean_architecture
import audit_notebook_and_model

REPORT_PATH = QA_DIR / "QA_AUDIT_REPORT.md"


def generate_markdown_report(suites: List[Dict[str, Any]]) -> str:
    total_checks = sum(s["total_count"] for s in suites)
    total_passed = sum(s["passed_count"] for s in suites)
    all_passed = total_passed == total_checks and total_checks > 0
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Reporte de Certificación QA y Validación de Arquitectura Limpia (SentinelCV)",
        "",
        f"- **Fecha de ejecución**: `{now_str}`",
        f"- **Resultado Global**: **{'✅ APROBADO (100%)' if all_passed else '⚠️ EN PROGRESO / PENDIENTE'}** (`{total_passed}/{total_checks}` controles superados)",
        "- **Auditor**: Agente 3 (`qa-validation-auditor`)",
        "",
        "## Resumen por Suite de Auditoría",
        "",
        "| Suite de Validación | Estado | Controles Superados |",
        "| :--- | :---: | :---: |",
    ]

    for s in suites:
        badge = "✅ PASS" if s["passed"] else "❌ FAIL / PENDING"
        lines.append(f"| **{s['suite']}** | {badge} | `{s['passed_count']}/{s['total_count']}` |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detalle de Controles Ejecutados")
    lines.append("")

    for s in suites:
        lines.append(f"### {s['suite']}")
        lines.append("")
        lines.append("| ID | Control | Resultado | Evidencia Técnica |")
        lines.append("| :--- | :--- | :---: | :--- |")
        for c in s["checks"]:
            status_icon = "✅ PASS" if c["passed"] else "❌ FAIL"
            details_clean = str(c["details"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{c['id']}` | {c['name']} | {status_icon} | {details_clean} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    syntax_only = "--syntax-check" in sys.argv

    if syntax_only:
        print(
            json.dumps(
                {
                    "status": "READY",
                    "message": "Los 5 scripts de auditoría en qa_validation/ fueron compilados e importados sin errores de sintaxis.",
                    "scripts": [
                        "audit_clean_architecture.py",
                        "audit_notebook_and_model.py",
                        "audit_api_contract.py",
                        "audit_astro_frontend.py",
                        "run_all_audits.py",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    suites = [
        audit_clean_architecture.run_audit(),
        audit_notebook_and_model.run_audit(),
        audit_api_contract.run_audit(),
        audit_astro_frontend.run_audit(),
    ]

    report_md = generate_markdown_report(suites)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    total_checks = sum(s["total_count"] for s in suites)
    total_passed = sum(s["passed_count"] for s in suites)
    all_passed = total_passed == total_checks and total_checks > 0

    summary = {
        "overall_passed": all_passed,
        "total_passed": total_passed,
        "total_checks": total_checks,
        "report_path": str(REPORT_PATH),
        "suites": suites,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

