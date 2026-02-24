"""
API REST para SAF - Predicción de humedad del suelo.

Permite enviar datos por JSON y recibir predicciones y capacitancias.
"""

import os
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import LinearModel, MLModel, CapacitanceDetector

# Caché de modelos Linear por sensor
_linear_model_cache: Dict[str, LinearModel] = {}

# Mínimo de puntos para entrenar el modelo Linear
MIN_TRAIN_POINTS = 500

WEIGHTS_DIR = "models/weights"


def get_weights_path(sensor_id: str) -> str:
    return f"{WEIGHTS_DIR}/sensor_{sensor_id}.joblib"


app = FastAPI(title="SAF API", description="API de predicción de humedad del suelo")


class DataPoint(BaseModel):
    date: str
    soil_moisture_40: float
    irrigation_volume_0: float = 0.0


class PredictRequest(BaseModel):
    data: List[DataPoint]
    model: str = Field(default="ML", description="ML o Linear")
    sensor_id: str = Field(default="", description="Obligatorio cuando model es Linear")
    previous_points: int = Field(default=10, ge=3)
    predict_steps: int = Field(default=100, ge=1)


class TrainRequest(BaseModel):
    sensor_id: str = Field(..., description="Identificador del sensor")
    model: str = Field(default="Linear", description="Solo Linear usa entrenamiento previo")
    data: List[DataPoint] = Field(..., description="Datos históricos del sensor")


class CapacitanceItem(BaseModel):
    date: str
    value: float


class PredictResponse(BaseModel):
    previous_values: List[float]
    predictions: List[float]
    capacitances: List[CapacitanceItem]
    dates: List[str]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas faltantes para compatibilidad con modelos y detector."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "soil_moisture_20" not in df.columns:
        df["soil_moisture_20"] = df["soil_moisture_40"]
    if "soil_moisture_60" not in df.columns:
        df["soil_moisture_60"] = df["soil_moisture_40"]
    if "irrigation_volume_accumulated_0" not in df.columns:
        df["irrigation_volume_accumulated_0"] = df["irrigation_volume_0"].cumsum()

    return df


@app.get("/health")
def health():
    """Comprueba que el servicio está activo."""
    return {"status": "ok"}


@app.get("/sensors/{sensor_id}/weights")
def get_weights_status(sensor_id: str):
    """
    Indica si el sensor tiene pesos entrenados para el modelo Linear.
    """
    path = get_weights_path(sensor_id)
    has_weights = os.path.exists(path)
    return {
        "sensor_id": sensor_id,
        "has_weights": has_weights,
        "path": path if has_weights else None,
    }


@app.post("/train")
def train(request: TrainRequest):
    """
    Entrena el modelo Linear con datos históricos del sensor y guarda los pesos.
    Los pesos se guardan en models/weights/sensor_{sensor_id}.joblib
    """
    if request.model != "Linear":
        raise HTTPException(
            status_code=400,
            detail="Solo el modelo Linear requiere entrenamiento previo. El modelo ML entrena en cada predicción.",
        )
    if not request.data:
        raise HTTPException(status_code=400, detail="data no puede estar vacío")
    if len(request.data) < MIN_TRAIN_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos {MIN_TRAIN_POINTS} puntos para entrenar. Se recibieron {len(request.data)}.",
        )

    raw = [
        {
            "date": p.date,
            "soil_moisture_40": p.soil_moisture_40,
            "irrigation_volume_0": p.irrigation_volume_0,
        }
        for p in request.data
    ]
    df = pd.DataFrame(raw)
    df = normalize_dataframe(df)
    df = df.sort_values("date").reset_index(drop=True)

    mask = df["irrigation_volume_0"] == 0
    if not mask.any():
        raise HTTPException(
            status_code=400,
            detail="Los datos deben incluir al menos un punto con irrigation_volume_0 == 0 (ciclos de riego-secado).",
        )

    try:
        model = LinearModel()
        model.train_plain_model(
            df.copy(),
            save_model=True,
            model_name=f"sensor_{request.sensor_id}",
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Datos insuficientes o formato incorrecto para entrenar: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}",
        )

    # Invalidar caché para que la próxima predict cargue el nuevo modelo
    _linear_model_cache.pop(request.sensor_id, None)

    weights_path = get_weights_path(request.sensor_id)
    return {
        "status": "ok",
        "sensor_id": request.sensor_id,
        "message": f"Modelo entrenado y guardado en {weights_path}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Recibe datos del sensor y devuelve predicciones y capacitancias.
    """
    if not request.data:
        raise HTTPException(status_code=400, detail="data no puede estar vacío")

    # Construir DataFrame
    raw = [
        {
            "date": p.date,
            "soil_moisture_40": p.soil_moisture_40,
            "irrigation_volume_0": p.irrigation_volume_0,
        }
        for p in request.data
    ]
    df = pd.DataFrame(raw)
    df = normalize_dataframe(df)
    df = df.sort_values("date").reset_index(drop=True)

    # Buscar primer punto con irrigation_volume_0 == 0 (inicio del decay)
    mask = df["irrigation_volume_0"] == 0
    if not mask.any():
        raise HTTPException(
            status_code=400,
            detail="Los datos deben incluir al menos un punto con irrigation_volume_0 == 0",
        )
    first_point_idx = mask.idxmax()
    df = df.loc[first_point_idx:].reset_index(drop=True)

    if len(df) < request.previous_points:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos {request.previous_points} puntos tras el riego, hay {len(df)}",
        )

    previous_values = df["soil_moisture_40"].iloc[: request.previous_points].values.tolist()
    current_date = df["date"].iloc[request.previous_points - 1]
    current_step = request.previous_points

    # Instanciar y usar modelo
    if request.model == "Linear":
        sensor_id = request.sensor_id.strip() if request.sensor_id else ""
        if not sensor_id:
            raise HTTPException(
                status_code=400,
                detail="sensor_id es obligatorio cuando se usa el modelo Linear",
            )
        model_path = get_weights_path(sensor_id)
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail=f"El sensor {sensor_id} no tiene archivo de pesos entrenados. Entrena el modelo primero con POST /train enviando datos históricos.",
            )
        if sensor_id not in _linear_model_cache:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            model = LinearModel()
            model.load_plain_model(model_path)
            _linear_model_cache[sensor_id] = model
        model = _linear_model_cache[sensor_id]
    elif request.model == "ML":
        model = MLModel()
        model.train(df.copy(), save_model=False)
    else:
        raise HTTPException(status_code=400, detail="model debe ser 'ML' o 'Linear'")

    predictions = model.predict_steps(
        previous_values, current_date, current_step, request.predict_steps
    )
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    else:
        predictions = list(predictions)

    # Detectar capacitancias
    capacitance_detector = CapacitanceDetector()
    try:
        capacitances_df = capacitance_detector.detect_capacitances(df)
    except ValueError:
        capacitances_df = pd.DataFrame(columns=["date", "capacitancy"])

    capacitances = [
        CapacitanceItem(date=str(row.date), value=float(row.capacitancy))
        for row in capacitances_df.itertuples(index=False)
    ]

    dates = [str(d) for d in df["date"].iloc[: request.previous_points]]

    return PredictResponse(
        previous_values=previous_values,
        predictions=predictions,
        capacitances=capacitances,
        dates=dates,
    )
