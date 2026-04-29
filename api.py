"""
API REST para SAF - Predicción de humedad del suelo.

Permite enviar datos por JSON y recibir predicciones y capacitancias.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Carga .env junto a este fichero (no hace falta repetir variables en cada terminal).
load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import LinearModel, MLModel, CapacitanceDetector
import weights_storage

# Caché de modelos Linear por sensor
_linear_model_cache: Dict[str, LinearModel] = {}
# Caché ML: sensor_id -> (clave S3 del .joblib cargado, modelo)
_ml_model_cache: Dict[str, Tuple[str, MLModel]] = {}

# Mínimo de puntos para entrenar Linear o ML (histórico)
MIN_TRAIN_POINTS = 500
MIN_ML_TRAIN_POINTS = 500

# Resolución entre valores consecutivos en `predictions` (alineada con `predict_steps` en Linear y ML).
PREDICTION_STEP_MINUTES = 30


def _s3_http_error(exc: ClientError) -> HTTPException:
    code = exc.response.get("Error", {}).get("Code", "Unknown")
    return HTTPException(
        status_code=503,
        detail=f"Error al acceder al almacenamiento de pesos (S3): {code}",
    )


def get_weights_path(sensor_id: str) -> str:
    """Clave de objeto S3 para el modelo Linear del sensor (mismo formato lógico que antes: prefix/sensor_id.joblib)."""
    return weights_storage.linear_object_key(sensor_id)


def build_ml_model_name(sensor_id: str) -> str:
    """Nombre base sin directorio ni extensión: ml_sensor_{id}_{YYYYMMDD_HHMMSS} (UTC)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"ml_sensor_{sensor_id}_{ts}"


def list_ml_weights_paths(sensor_id: str) -> List[str]:
    """Claves S3 de artefactos ML del sensor, más reciente primero (orden lexicográfico del sufijo)."""
    try:
        return weights_storage.list_ml_object_keys(sensor_id)
    except ClientError as e:
        raise _s3_http_error(e) from e


def get_latest_ml_weights_path(sensor_id: str) -> Optional[str]:
    paths = list_ml_weights_paths(sensor_id)
    return paths[0] if paths else None


def get_cached_ml_model(sensor_id: str, weights_object_key: str) -> MLModel:
    """Carga o reutiliza desde caché el modelo ML descargando el .joblib desde S3 si hace falta."""
    cached = _ml_model_cache.get(sensor_id)
    if cached is None or cached[0] != weights_object_key:
        tmp: Optional[str] = None
        try:
            tmp = weights_storage.download_to_tempfile(weights_object_key)
            model = MLModel()
            model.load_model(tmp)
        except ClientError as e:
            raise _s3_http_error(e) from e
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        _ml_model_cache[sensor_id] = (weights_object_key, model)
    return _ml_model_cache[sensor_id][1]


def build_prediction_dates(anchor: pd.Timestamp, n: int) -> List[str]:
    """Fechas de cada valor en `predictions`: t_i = anchor + (i+1) * PREDICTION_STEP_MINUTES."""
    if n <= 0:
        return []
    step = pd.Timedelta(minutes=PREDICTION_STEP_MINUTES)
    return [str(anchor + step * (i + 1)) for i in range(n)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if weights_storage.skip_startup_validation():
        print(
            "SAF: aviso — no se validó S3 al arrancar (S3_SKIP_STARTUP_VALIDATION). "
            "/train y /predict pueden fallar si el bucket o credenciales no son correctos.",
            flush=True,
        )
    else:
        weights_storage.validate_bucket_access()
        print(
            f"SAF: almacén de pesos S3 accesible "
            f"(bucket={weights_storage.bucket_name()}, región={weights_storage.region_name()})",
            flush=True,
        )
    yield


app = FastAPI(
    title="SAF API",
    description="API de predicción de humedad del suelo",
    lifespan=lifespan,
)


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
    Los ficheros residen en S3; path es la clave de objeto (mismo formato lógico que en despliegues anteriores).
    """
    try:
        linear_path = get_weights_path(sensor_id)
        linear_exists = weights_storage.object_exists(linear_path)
        ml_paths = list_ml_weights_paths(sensor_id)
    except ClientError as e:
        raise _s3_http_error(e) from e
    latest_ml = ml_paths[0] if ml_paths else None

    return {
        "sensor_id": sensor_id,
        "storage": {"backend": "s3", "bucket": weights_storage.bucket_name()},
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
    Entrena Linear o ML con datos históricos y sube los .joblib al bucket S3 configurado.
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
        weights_path = get_weights_path(request.sensor_id)
        tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            try:
                model = LinearModel()
                model.train_plain_model(
                    df.copy(),
                    save_model=True,
                    model_name=f"sensor_{request.sensor_id}",
                    output_path=tmp_path,
                )
            except HTTPException:
                raise
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
            try:
                weights_storage.upload_file(tmp_path, weights_path)
            except ClientError as e:
                raise _s3_http_error(e) from e
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        _linear_model_cache.pop(request.sensor_id, None)
        return {
            "status": "ok",
            "sensor_id": request.sensor_id,
            "model": "Linear",
            "path": weights_path,
            "message": f"Modelo Linear guardado en s3://{weights_storage.bucket_name()}/{weights_path}",
        }

    # ML
    model_name = build_ml_model_name(request.sensor_id)
    weights_path = weights_storage.ml_object_key(model_name)
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        try:
            ml = MLModel()
            ml.train(df.copy(), save_model=True, model_name=model_name, output_path=tmp_path)
        except HTTPException:
            raise
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
        try:
            weights_storage.upload_file(tmp_path, weights_path)
        except ClientError as e:
            raise _s3_http_error(e) from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    _ml_model_cache.pop(request.sensor_id, None)
    return {
        "status": "ok",
        "sensor_id": request.sensor_id,
        "model": "ML",
        "path": weights_path,
        "message": f"Modelo ML guardado en s3://{weights_storage.bucket_name()}/{weights_path}",
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
        try:
            has_linear = weights_storage.object_exists(model_path)
        except ClientError as e:
            raise _s3_http_error(e) from e
        if not has_linear:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"El sensor {sensor_id} no tiene archivo de pesos Linear. "
                    f"Entrena con POST /train (model: \"Linear\", mínimo {MIN_TRAIN_POINTS} puntos). "
                    f"Clave S3 esperada: {model_path}"
                ),
            )
        if sensor_id not in _linear_model_cache:
            tmp: Optional[str] = None
            try:
                tmp = weights_storage.download_to_tempfile(model_path)
                model = LinearModel()
                model.load_plain_model(tmp)
            except ClientError as e:
                raise _s3_http_error(e) from e
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
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
                    f"Se creará un objeto bajo {weights_storage.weights_prefix()}/ml_sensor_{sensor_id}_YYYYMMDD_HHMMSS.joblib (UTC) en S3."
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
