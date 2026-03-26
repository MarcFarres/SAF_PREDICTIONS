"""
API REST para SAF - Predicción de humedad del suelo.

Permite enviar datos por JSON y recibir predicciones y capacitancias.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import LinearModel, MLModel, CapacitanceDetector

# Caché de modelos Linear por sensor
_linear_model_cache: Dict[str, LinearModel] = {}
# Caché ML: sensor_id -> (ruta del .joblib cargada, modelo)
_ml_model_cache: Dict[str, Tuple[str, MLModel]] = {}

# Mínimo de puntos para entrenar Linear o ML (histórico)
MIN_TRAIN_POINTS = 500
MIN_ML_TRAIN_POINTS = 500

# Resolución entre valores consecutivos en `predictions` (alineada con `predict_steps` en Linear y ML).
PREDICTION_STEP_MINUTES = 30

WEIGHTS_DIR = "models/weights"


def get_weights_path(sensor_id: str) -> str:
    return f"{WEIGHTS_DIR}/sensor_{sensor_id}.joblib"


def build_ml_model_name(sensor_id: str) -> str:
    """Nombre base sin directorio ni extensión: ml_sensor_{id}_{YYYYMMDD_HHMMSS} (UTC)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"ml_sensor_{sensor_id}_{ts}"


def list_ml_weights_paths(sensor_id: str) -> List[str]:
    """Rutas absolutas relativas al cwd, más reciente primero (orden lexicográfico del sufijo)."""
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    prefix = f"ml_sensor_{sensor_id}_"
    names = [
        n
        for n in os.listdir(WEIGHTS_DIR)
        if n.startswith(prefix) and n.endswith(".joblib")
    ]
    names.sort(reverse=True)
    return [os.path.join(WEIGHTS_DIR, n) for n in names]


def get_latest_ml_weights_path(sensor_id: str) -> Optional[str]:
    paths = list_ml_weights_paths(sensor_id)
    return paths[0] if paths else None


def get_cached_ml_model(sensor_id: str, weights_path: str) -> MLModel:
    """Carga o reutiliza desde caché el modelo ML desde `weights_path` (ruta al .joblib más reciente)."""
    cached = _ml_model_cache.get(sensor_id)
    if cached is None or cached[0] != weights_path:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model = MLModel()
        model.load_model(weights_path)
        _ml_model_cache[sensor_id] = (weights_path, model)
    return _ml_model_cache[sensor_id][1]


def build_prediction_dates(anchor: pd.Timestamp, n: int) -> List[str]:
    """Fechas de cada valor en `predictions`: t_i = anchor + (i+1) * PREDICTION_STEP_MINUTES."""
    if n <= 0:
        return []
    step = pd.Timedelta(minutes=PREDICTION_STEP_MINUTES)
    return [str(anchor + step * (i + 1)) for i in range(n)]


app = FastAPI(title="SAF API", description="API de predicción de humedad del suelo")


class DataPoint(BaseModel):
    date: str
    soil_moisture_40: float
    irrigation_volume_0: float = 0.0


class PredictRequest(BaseModel):
    data: List[DataPoint]
    model: str = Field(default="ML", description="ML o Linear")
    sensor_id: str = Field(
        default="",
        description="Obligatorio para modelos ML y Linear",
    )
    previous_points: int = Field(default=10, ge=3)
    predict_steps: int = Field(default=100, ge=1)


class TrainRequest(BaseModel):
    sensor_id: str = Field(..., description="Identificador del sensor")
    model: str = Field(default="Linear", description='"Linear" o "ML"')
    data: List[DataPoint] = Field(..., description="Datos históricos del sensor")


class CapacitanceItem(BaseModel):
    date: str
    value: float


class PredictResponse(BaseModel):
    previous_values: List[float]
    predictions: List[float]
    prediction_dates: List[str]
    capacitances: List[CapacitanceItem]
    dates: List[str]
    ccpmp: float  # Umbral de humedad sugerido según predicción (mínimo esperado)


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


def _train_dataframe_from_request(request: TrainRequest) -> pd.DataFrame:
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
    return df


@app.get("/health")
def health():
    """Comprueba que el servicio está activo."""
    return {"status": "ok"}


@app.get("/sensors/{sensor_id}/weights")
def get_weights_status(sensor_id: str):
    """
    Estado de pesos entrenados por sensor: Linear (sensor_{id}.joblib) y ML (ml_sensor_{id}_*.joblib).
    """
    linear_path = get_weights_path(sensor_id)
    linear_exists = os.path.exists(linear_path)
    ml_paths = list_ml_weights_paths(sensor_id)
    latest_ml = ml_paths[0] if ml_paths else None

    return {
        "sensor_id": sensor_id,
        "linear": {
            "has_weights": linear_exists,
            "path": linear_path if linear_exists else None,
        },
        "ml": {
            "has_weights": len(ml_paths) > 0,
            "latest_path": latest_ml,
            "all_paths": ml_paths,
        },
    }


@app.post("/train")
def train(request: TrainRequest):
    """
    Entrena Linear o ML con datos históricos y guarda pesos bajo models/weights/.
    """
    if request.model not in ("Linear", "ML"):
        raise HTTPException(
            status_code=400,
            detail='model debe ser "Linear" o "ML".',
        )
    if not request.data:
        raise HTTPException(status_code=400, detail="data no puede estar vacío")

    min_pts = MIN_TRAIN_POINTS if request.model == "Linear" else MIN_ML_TRAIN_POINTS
    if len(request.data) < min_pts:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos {min_pts} puntos para entrenar {request.model}. Se recibieron {len(request.data)}.",
        )

    df = _train_dataframe_from_request(request)

    if request.model == "Linear":
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
        _linear_model_cache.pop(request.sensor_id, None)
        weights_path = get_weights_path(request.sensor_id)
        return {
            "status": "ok",
            "sensor_id": request.sensor_id,
            "model": "Linear",
            "path": weights_path,
            "message": f"Modelo Linear guardado en {weights_path}",
        }

    # ML
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_name = build_ml_model_name(request.sensor_id)
    try:
        ml = MLModel()
        ml.train(df.copy(), save_model=True, model_name=model_name)
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Datos insuficientes para entrenar ML (pocas filas de decay): {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento ML: {str(e)}",
        )
    weights_path = f"{WEIGHTS_DIR}/{model_name}.joblib"
    _ml_model_cache.pop(request.sensor_id, None)
    return {
        "status": "ok",
        "sensor_id": request.sensor_id,
        "model": "ML",
        "path": weights_path,
        "message": f"Modelo ML guardado en {weights_path}",
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
                detail=(
                    f"El sensor {sensor_id} no tiene archivo de pesos Linear. "
                    f"Entrena con POST /train (model: \"Linear\", mínimo {MIN_TRAIN_POINTS} puntos). "
                    f"Ruta esperada: {model_path}"
                ),
            )
        if sensor_id not in _linear_model_cache:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            model = LinearModel()
            model.load_plain_model(model_path)
            _linear_model_cache[sensor_id] = model
        model = _linear_model_cache[sensor_id]
    elif request.model == "ML":
        sensor_id = request.sensor_id.strip() if request.sensor_id else ""
        if not sensor_id:
            raise HTTPException(
                status_code=400,
                detail="sensor_id es obligatorio cuando se usa el modelo ML",
            )
        latest = get_latest_ml_weights_path(sensor_id)
        if latest is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"El sensor {sensor_id} no tiene modelo ML entrenado. "
                    f"Entrena con POST /train enviando JSON con \"sensor_id\", \"model\": \"ML\" y \"data\" "
                    f"(mínimo {MIN_ML_TRAIN_POINTS} puntos con ciclos de riego-secado). "
                    f"Se creará un archivo bajo {WEIGHTS_DIR}/ml_sensor_{sensor_id}_YYYYMMDD_HHMMSS.joblib (UTC)."
                ),
            )
        model = get_cached_ml_model(sensor_id, latest)
    else:
        raise HTTPException(status_code=400, detail="model debe ser 'ML' o 'Linear'")

    predictions = model.predict_steps(
        previous_values, current_date, current_step, request.predict_steps
    )
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    else:
        predictions = list(predictions)

    prediction_dates = build_prediction_dates(current_date, len(predictions))

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

    ccpmp = min(predictions) if predictions else previous_values[-1]

    return PredictResponse(
        previous_values=previous_values,
        predictions=predictions,
        prediction_dates=prediction_dates,
        capacitances=capacitances,
        dates=dates,
        ccpmp=ccpmp,
    )
