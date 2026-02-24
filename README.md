# SAF

Sistema de **predicción y análisis de humedad del suelo** para riego agrícola. Usa datos de sensores para predecir la evolución de la humedad, detectar momentos de capacitancia y ayudar a decidir cuándo regar.

El informe con los resultados e insights del proyecto se encuentra [aquí](SAF_report.pdf).

## Instalación

```bash
pip install -r requirements.txt
```

## Cómo usar el proyecto

### 1. Preparar los datos (si tienes un CSV crudo SAF)

Si tienes un archivo CSV en formato SAF original, primero hay que corregirlo:

```bash
python preprocess_csv.py --path data/tu-archivo.csv
```

Esto genera `tu-archivo-Fix.csv` en la misma carpeta.

### 2. Ejecutar el programa principal

```bash
# Desde la carpeta del proyecto, ejecutar (usa modelo ML por defecto)
python main.py

# Elegir modelo explícitamente
python main.py --model ML      # modelo de Machine Learning (XGBoost)
python main.py -m Linear       # modelo lineal

# Ver ayuda
python main.py --help
```

**Nota:** El script espera el archivo `data/1082-Device-Data-Fix.csv`. Si usas otro archivo, tendrás que modificar la ruta en `main.py`.

### 3. Resultados

Al ejecutar `main.py` se muestran dos gráficas:

1. **Predicción vs real:** puntos azules (valores vistos), línea roja (predicciones), línea verde (valores reales).
2. **Capacitancias:** humedad en el tiempo con líneas verticales rojas marcando los momentos de capacitancia (cuando el suelo deja de drenar y se estabiliza).

## API REST

El proyecto expone una API HTTP para integrarse con otros sistemas (por ejemplo Symfony/PHP).

### Arrancar la API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

La documentación interactiva (Swagger) estará en `http://localhost:8000/docs`.

### Endpoints

| Método | Ruta      | Descripción                |
|--------|-----------|----------------------------|
| GET    | /health   | Comprueba que el servicio está activo |
| GET    | /sensors/{sensor_id}/weights | Indica si el sensor tiene pesos entrenados |
| POST   | /train   | Entrena el modelo Linear con datos históricos del sensor |
| POST   | /predict  | Recibe datos y devuelve predicciones y capacitancias |

### POST /train (modelo Linear)

Entrena el modelo Linear con datos históricos y guarda los pesos en `models/weights/sensor_{sensor_id}.joblib`. Requiere al menos 500 puntos con ciclos de riego-secado.

```json
{
  "sensor_id": "1082",
  "model": "Linear",
  "data": [
    {"date": "2024-01-15 08:00:00", "soil_moisture_40": 0.35, "irrigation_volume_0": 10},
    {"date": "2024-01-15 08:30:00", "soil_moisture_40": 0.34, "irrigation_volume_0": 0},
    ...
  ]
}
```

### POST /predict

```json
{
  "data": [
    {"date": "2024-06-06 12:00:00", "soil_moisture_40": 0.32, "irrigation_volume_0": 0},
    {"date": "2024-06-06 12:30:00", "soil_moisture_40": 0.30, "irrigation_volume_0": 0}
  ],
  "model": "ML",
  "sensor_id": "1082",
  "previous_points": 10,
  "predict_steps": 100
}
```

- `data`: array de puntos con `date`, `soil_moisture_40`, `irrigation_volume_0` (obligatorios).
- `model`: `"ML"` o `"Linear"` (por defecto `"ML"`). Con Linear, los pesos se cargan de `models/weights/sensor_{sensor_id}.joblib`.
- `sensor_id`: obligatorio cuando `model` es `"Linear"`. Si el sensor no tiene pesos entrenados, la API devuelve 503 con el mensaje correspondiente.
- `previous_points`: puntos previos para la predicción (por defecto 10, mínimo 3).
- `predict_steps`: pasos futuros a predecir (por defecto 100).

### Ejemplo de respuesta

```json
{
  "previous_values": [0.32, 0.30, ...],
  "predictions": [0.28, 0.27, ...],
  "capacitances": [{"date": "2024-06-08T10:00:00", "value": 0.25}, ...],
  "dates": ["2024-06-06 12:00:00", "2024-06-06 12:30:00", ...]
}
```

## Estructura del repositorio

```
SAF/
├── logger                          <- Formateador de logs
├── models                          <- Clases y pesos de los modelos
│   └── weights                     <- Modelos guardados
├── notebooks                       <- Notebooks de desarrollo (mover a raíz para que funcionen)
├── plots                           <- Gráficas obtenidas durante el desarrollo
│   ├── animations                  <- Animaciones de visualización
│   ├── errors                      <- Gráficas de evolución de errores
│   └── preds                       <- Gráficas de predicción de ejemplo
├── results                         <- Errores extraídos de los modelos
├── trainers                        <- Deprecado. Solo para algunos notebooks
├── utils                           <- Funciones auxiliares compartidas
├── compute_mean_losses_linear.py   <- Script para calcular pérdidas del modelo lineal
├── compute_mean_losses_ML.py       <- Script para calcular pérdidas del modelo ML
├── api.py                          <- API REST (FastAPI) para predicción por HTTP
├── preprocess_csv.py               <- Script para limpiar CSVs SAF y que pandas los lea
└── main.py                         <- Ejemplo de uso de las clases
```

Los datos deben estar en una carpeta llamada `data`.

## Formato del CSV crudo (preprocess_csv)

El CSV que se pasa a `preprocess_csv.py` debe cumplir lo siguiente:

### Estructura de las columnas

- **Columna 1 (`day`):** valor antes del primer `,`
- **Columna 2 (`date`):** valor entre el primer y el segundo `,`
- **Resto (`data`):** el resto de columnas

El script concatena `day` y `date` (sin espacio) para formar la columna `date` en el archivo Fix. Esa fecha final debe poder parsearse como `"%b %d %Y @ %H:%M:%S.%f"` (ej: `Jun 06 2024 @ 12:00:00.000`).

### Columnas necesarias en la parte `data`

| Columna                     | Descripción                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| `variable.name`             | Nombre de la variable: `soil_moisture`, `irrigation_volume`, etc.            |
| `depth`                     | Profundidad en cm (20, 40, 60 para humedad; 0 para riego)                     |
| `variable.normalized_value` | Valor normalizado de la medición (float)                                     |
| `sensor.deviceSensorid`     | Se elimina tras el Fix                                                        |
| `position`                  | Se elimina tras el Fix                                                        |
| `sensor.idDecagon`          | Se elimina tras el Fix                                                        |
| `variable.default_value_name` | Se elimina tras el Fix                                                     |

### Formato largo

Una fila por combinación de (fecha, variable, profundidad). Ejemplo conceptual para un mismo timestamp:

```
day, date, sensor.deviceSensorid, position, ..., variable.name, depth, variable.normalized_value, ...
Fri, Jun 06 2024 @ 12:00:00.000, ..., soil_moisture, 40, 0.28, ...
Fri, Jun 06 2024 @ 12:00:00.000, ..., soil_moisture, 20, 0.25, ...
Fri, Jun 06 2024 @ 12:00:00.000, ..., irrigation_volume, 0, 0, ...
```

### Nota sobre comillas

`preprocess_csv.py` elimina todas las comillas del archivo de salida, tanto en la cabecera como en los datos.

## Datos de entrada esperados

Los DataFrames de entrada deben tener al menos estas columnas tras el preprocesamiento (`get_clean_df` sobre el Fix):

| Columna             | Tipo              | Descripción                                      |
|---------------------|-------------------|--------------------------------------------------|
| `date`              | `pd.Timestamp`    | Timestamp de la medición de humedad              |
| `soil_moisture_40`  | `float`          | Humedad del suelo a profundidad 40 cm           |

Ambos modelos y el detector de capacitancia usan la humedad a 40 cm, que es la que mejor comportamiento ha mostrado. Para usar otras profundidades habría que adaptar el entrenamiento; otra opción es renombrar la columna de humedad deseada a `soil_moisture_40`.

También se requiere `irrigation_volume_0` para la lógica de riego y detección de capacitancias.

## Modelos disponibles

- **ML:** XGBRegressor, entrena rápido, predice en bloques de 3 pasos (90 min).
- **Linear:** Regresión lineal (RANSAC) + detector de mesetas (XGBClassifier). Entrena más lento; se recomienda entrenar una vez y cargar el modelo guardado.
