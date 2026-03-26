"""
Genera JSON de ejemplo para probar la API SAF (train / predict).
Ejecutar desde la raíz del proyecto: python insomnia/generate_payloads.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "payloads"


def synthetic_series(num_points: int = 520) -> list[dict]:
    """Serie cada 30 min con riego inicial y ciclos ocasionales; decay de humedad en tramos sin riego."""
    base = datetime(2024, 6, 1, 8, 0, 0)
    data: list[dict] = []
    moisture = 0.42
    for i in range(num_points):
        t = base + timedelta(minutes=30 * i)
        if i == 0:
            irr = 15.0
        elif i % 100 == 0:
            irr = 10.0
            moisture = min(0.44, moisture + 0.06)
        else:
            irr = 0.0
        if irr == 0:
            moisture = max(0.12, moisture * 0.9975 - 0.00015)
        data.append(
            {
                "date": t.strftime("%Y-%m-%d %H:%M:%S"),
                "soil_moisture_40": round(moisture, 5),
                "irrigation_volume_0": irr,
            }
        )
    return data


def predict_short_series() -> list[dict]:
    """Ventana corta post-riego para POST /predict (>= previous_points tras el corte)."""
    base = datetime(2024, 6, 10, 6, 0, 0)
    data = []
    # Un punto con riego, luego solo secado
    data.append(
        {
            "date": base.strftime("%Y-%m-%d %H:%M:%S"),
            "soil_moisture_40": 0.38,
            "irrigation_volume_0": 12.0,
        }
    )
    moisture = 0.37
    for i in range(1, 20):
        t = base + timedelta(minutes=30 * i)
        moisture = max(0.22, moisture * 0.992)
        data.append(
            {
                "date": t.strftime("%Y-%m-%d %H:%M:%S"),
                "soil_moisture_40": round(moisture, 5),
                "irrigation_volume_0": 0.0,
            }
        )
    return data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sensor_id = "1082"
    train_data = synthetic_series(520)

    train_ml = {
        "sensor_id": sensor_id,
        "model": "ML",
        "data": train_data,
    }
    train_linear = {
        "sensor_id": sensor_id,
        "model": "Linear",
        "data": train_data,
    }
    predict_ml = {
        "data": predict_short_series(),
        "model": "ML",
        "sensor_id": sensor_id,
        "previous_points": 10,
        "predict_steps": 48,
    }
    predict_linear = {
        **predict_ml,
        "model": "Linear",
    }

    (OUT / "train_ml.json").write_text(
        json.dumps(train_ml, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "train_linear.json").write_text(
        json.dumps(train_linear, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "predict_ml.json").write_text(
        json.dumps(predict_ml, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "predict_linear.json").write_text(
        json.dumps(predict_linear, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Escritos en {OUT}")


if __name__ == "__main__":
    main()
