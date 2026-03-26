"""
Construye SAF_API.insomnia.json (Insomnia export format 4) incrustando payloads.
Ejecutar tras generate_payloads.py: python insomnia/build_insomnia_export.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PAYLOADS = ROOT / "payloads"
EXPORT_PATH = ROOT / "SAF_API.insomnia.json"

BASE_URL = "https://f26d-54-73-118-61.ngrok-free.app"

NGROK_HEADERS = [
    {"id": "h_ngrok", "name": "ngrok-skip-browser-warning", "value": "true"},
]


def load_payload(name: str) -> str:
    return (PAYLOADS / name).read_text(encoding="utf-8")


def req(
    _id: str,
    parent: str,
    name: str,
    method: str,
    path: str,
    body_text: str | None = None,
    sort_key: float = 0,
) -> dict:
    body: dict = {}
    if body_text is not None:
        body = {"mimeType": "application/json", "text": body_text}
    return {
        "_id": _id,
        "parentId": parent,
        "_type": "request",
        "name": name,
        "method": method,
        "url": "{{ _.base_url }}" + path,
        "headers": NGROK_HEADERS,
        "body": body,
        "parameters": [],
        "authentication": {},
        "metaSortKey": sort_key,
        "isPrivate": False,
        "settingStoreCookies": True,
        "settingSendCookies": True,
        "settingDisableRenderRequestBody": False,
        "settingEncodeUrl": True,
        "settingRebuildPath": True,
        "settingFollowRedirects": "global",
    }


def main() -> None:
    if not (PAYLOADS / "train_ml.json").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "generate_payloads.py")],
            check=True,
            cwd=str(ROOT.parent),
        )

    wrk = "wrk_saf_api"
    fld = "fld_saf_requests"
    env = "env_saf_ngrok"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    resources = [
        {
            "_id": wrk,
            "_type": "workspace",
            "name": "SAF API (ngrok)",
            "description": "Pruebas contra la API SAF vía ngrok.",
        },
        {
            "_id": env,
            "_type": "environment",
            "parentId": wrk,
            "name": "ngrok",
            "data": {"base_url": BASE_URL},
            "dataPropertyOrder": {"&": ["base_url"]},
            "color": "#7d69cb",
            "isPrivate": False,
            "metaSortKey": 1,
        },
        {
            "_id": fld,
            "_type": "request_group",
            "parentId": wrk,
            "name": "Endpoints",
            "environment": {},
            "environmentPropertyOrder": None,
            "metaSortKey": -1000,
        },
        req("req_health", fld, "01 GET /health", "GET", "/health", None, -3000),
        req(
            "req_weights",
            fld,
            "02 GET /sensors/1082/weights",
            "GET",
            "/sensors/1082/weights",
            None,
            -2500,
        ),
        req(
            "req_train_ml",
            fld,
            "03 POST /train (ML, ~520 puntos)",
            "POST",
            "/train",
            load_payload("train_ml.json"),
            -2000,
        ),
        req(
            "req_train_linear",
            fld,
            "04 POST /train (Linear, ~520 puntos)",
            "POST",
            "/train",
            load_payload("train_linear.json"),
            -1500,
        ),
        req(
            "req_predict_ml",
            fld,
            "05 POST /predict (ML)",
            "POST",
            "/predict",
            load_payload("predict_ml.json"),
            -1000,
        ),
        req(
            "req_predict_linear",
            fld,
            "06 POST /predict (Linear)",
            "POST",
            "/predict",
            load_payload("predict_linear.json"),
            -500,
        ),
    ]

    export = {
        "_type": "export",
        "__export_format": 4,
        "__export_date": now,
        "__export_source": "insomnia.desktop.app",
        "resources": resources,
    }

    EXPORT_PATH.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Escrito {EXPORT_PATH}")


if __name__ == "__main__":
    main()
