"""Persistencia de artefactos .joblib en Amazon S3.

Configuración por variables de entorno:
- S3_WEIGHTS_BUCKET: nombre del bucket (por defecto ag-zoom-sensor-ml-eu-west-1)
- AWS_REGION o AWS_DEFAULT_REGION: región (por defecto eu-west-1)
- S3_WEIGHTS_PREFIX: prefijo de claves sin barra inicial/final (por defecto models/weights)
- S3_SKIP_STARTUP_VALIDATION: si es 1/true/yes, no se llama a HeadBucket al arrancar (solo para desarrollo; train/predict seguirán necesitando S3)

Credenciales: cadena estándar de boto3 (perfil, variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, rol IAM, etc.).
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import boto3
from botocore.exceptions import ClientError

_DEFAULT_BUCKET = "ag-zoom-sensor-ml-eu-west-1"
_DEFAULT_REGION = "eu-west-1"


def bucket_name() -> str:
    return os.environ.get("S3_WEIGHTS_BUCKET", _DEFAULT_BUCKET).strip()


def region_name() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or _DEFAULT_REGION
    )


def weights_prefix() -> str:
    """Prefijo de claves S3, p.ej. models/weights (sin slashes sobrantes)."""
    return (os.environ.get("S3_WEIGHTS_PREFIX") or "models/weights").strip().strip("/")


def linear_object_key(sensor_id: str) -> str:
    p = weights_prefix()
    return f"{p}/sensor_{sensor_id}.joblib"


def ml_object_key(model_name: str) -> str:
    p = weights_prefix()
    return f"{p}/{model_name}.joblib"


def ml_list_prefix(sensor_id: str) -> str:
    p = weights_prefix()
    return f"{p}/ml_sensor_{sensor_id}_"


_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=region_name())
    return _client


def skip_startup_validation() -> bool:
    v = (os.environ.get("S3_SKIP_STARTUP_VALIDATION") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _buckets_visible_hint(c) -> str:
    """Ayuda a diagnosticar nombre de bucket / cuenta; requiere s3:ListAllMyBuckets."""
    try:
        resp = c.list_buckets()
        names = sorted(b["Name"] for b in resp.get("Buckets", []))
        if not names:
            return (
                " Con estas credenciales la lista de buckets está vacía (cuenta sin buckets "
                "o falta permiso s3:ListAllMyBuckets)."
            )
        max_show = 40
        tail = f" (+{len(names) - max_show} más)" if len(names) > max_show else ""
        return (
            f" Buckets visibles con estas credenciales: "
            f"{', '.join(names[:max_show])}{tail}."
        )
    except ClientError as le:
        ccode = le.response.get("Error", {}).get("Code", "")
        return f" No se pudo listar buckets (¿falta s3:ListAllMyBuckets?): {ccode}."
    except Exception:
        return ""


def validate_bucket_access() -> None:
    """Comprueba que el bucket existe y hay permisos. Usar al arranque de la aplicación."""
    c = get_client()
    b = bucket_name()
    reg = region_name()
    try:
        c.head_bucket(Bucket=b)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", "") or "")
        msg = e.response.get("Error", {}).get("Message", str(e))
        hint = ""
        if code in ("404", "NotFound") or "404" in str(e):
            hint = (
                " Comprueba en la consola AWS el nombre exacto del bucket y la región; "
                "copia S3_WEIGHTS_BUCKET sin espacios ni comillas en .env. "
                "Las credenciales deben ser de la cuenta donde existe el bucket."
            )
            hint += _buckets_visible_hint(c)
            hint += " Para arrancar sin esta comprobación (solo dev): S3_SKIP_STARTUP_VALIDATION=1."
        elif code == "301" or "PermanentRedirect" in code:
            hint = (
                " El bucket puede estar en otra región: revisa en S3 la región del bucket "
                "y pon AWS_REGION con ese valor."
            )
        elif code in ("403", "Forbidden"):
            hint = (
                " La clave IAM no tiene acceso a este bucket o está en otra cuenta."
                + _buckets_visible_hint(c)
            )

        raise RuntimeError(
            f"No se pudo acceder al bucket S3 '{b}' (región cliente boto3: {reg}). "
            f"Código: {code or msg}.{hint}"
        ) from e


def object_exists(key: str) -> bool:
    try:
        get_client().head_object(Bucket=bucket_name(), Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def list_ml_object_keys(sensor_id: str) -> List[str]:
    prefix = ml_list_prefix(sensor_id)
    keys: List[str] = []
    paginator = get_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name(), Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj["Key"]
            if k.endswith(".joblib"):
                keys.append(k)
    keys.sort(reverse=True)
    return keys


def upload_file(local_path: str, key: str) -> None:
    get_client().upload_file(local_path, bucket_name(), key)


def download_to_tempfile(key: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    try:
        get_client().download_file(bucket_name(), key, path)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path
