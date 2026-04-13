# Resumen: integración de `/predict` (frontend → API SAF)

Este documento resume el trabajo realizado para que una aplicación web pueda obtener predicciones de humedad llamando a **`POST /predict`** del servicio Python (FastAPI en `api.py`), y sirve de puente hacia el creador del algoritmo sobre **qué se envía, por qué y qué se recibe**.

---

## 1. Qué se ha hecho a nivel de producto

- El núcleo SAF (modelos `MLModel`, `LinearModel`, detector `CapacitanceDetector`) se expone como **API HTTP** (`uvicorn api:app`), en lugar de ejecutar solo `main.py` en local.
- El **frontend no llama normalmente a Python en bruto**: el flujo previsto es **Frontend → backend (p. ej. Symfony) → API Python** (`docs/integracion.md`). El backend reúne datos del sensor (p. ej. Elasticsearch), los **mapea al JSON que exige la API** y devuelve al frontend la respuesta ya lista para gráficos.
- La API valida el cuerpo con **Pydantic**, ordena las filas por fecha, aplica la misma lógica de “ventana tras el riego” que el script de referencia, entrena o carga el modelo, ejecuta `predict_steps`, calcula capacitancias y devuelve un JSON estructurado.

---

## 2. Formato del cuerpo JSON (lo que debe enviar el cliente)

Todo va en **`Content-Type: application/json`**. Estructura del **request** (`PredictRequest` en `api.py`):

| Campo | Tipo | Obligatorio / default | Formato esperado |
|--------|------|------------------------|------------------|
| `data` | array | **Obligatorio**, no vacío | Lista de objetos **punto a punto** (ver tabla siguiente) |
| `model` | string | Default `"ML"` | `"ML"` o `"Linear"` |
| `sensor_id` | string | Default `""` | **Obligatorio si `model` es `"Linear"`** (identifica el archivo de pesos del sensor) |
| `previous_points` | entero | Default `10`, **mínimo 3** | Cuántos puntos consecutivos de humedad se usan como “historia” previa a la predicción |
| `predict_steps` | entero | Default `100`, mínimo `1` | Cuántos pasos futuros se predicen |

Cada elemento de `data` debe ser un objeto con:

| Campo | Tipo | Notas |
|--------|------|--------|
| `date` | string | Cualquier formato que **pandas pueda parsear** como fecha-hora (p. ej. `"2024-06-06 12:00:00"` o ISO 8601). La API convierte con `pd.to_datetime`. |
| `soil_moisture_40` | número (float) | Humedad del suelo a 40 cm, en la misma escala que usó el entrenamiento (típicamente 0–1 o fracción). |
| `irrigation_volume_0` | número (float) | Default `0` si no se envía. Volumen de riego en el instante del punto (0 = sin riego en ese muestreo). |

**Ejemplo mínimo válido (estructura):**

```json
{
  "data": [
    {"date": "2024-06-06 12:00:00", "soil_moisture_40": 0.32, "irrigation_volume_0": 0},
    {"date": "2024-06-06 12:30:00", "soil_moisture_40": 0.30, "irrigation_volume_0": 0}
  ],
  "model": "ML",
  "sensor_id": "",
  "previous_points": 10,
  "predict_steps": 100
}
```

**Qué datos estamos esperando recibir en la práctica:** una **serie temporal ordenable** de mediciones del mismo sensor, con las tres magnitudes anteriores. Si el origen es otro esquema (Elasticsearch, etc.), el backend debe **transformar** nombres de campos y tipos para cumplir exactamente este contrato (véase la sección de mapeo en `docs/integracion.md`).

---

## 3. Por qué cada parámetro (según el código que se ejecuta)

La lógica está en `predict()` en `api.py` (aprox. líneas 167–265).

- **`data` (y cada `date`, `soil_moisture_40`, `irrigation_volume_0`)**  
  - Se construye un `DataFrame`, se normaliza (`normalize_dataframe`: fechas, humedades duplicadas a otras profundidades si faltan, acumulado de riego) y se **ordena por `date`**.  
  - Se exige **al menos un punto con `irrigation_volume_0 == 0`**: con eso se localiza el **primer índice tras el riego** (`mask.idxmax()`), y la serie se recorta **desde ese punto** (`df.loc[first_point_idx:]`). Esto replica la idea de `main.py`: trabajar en la fase de **decaimiento** tras el riego, no mezclando arbitrariamente todo el histórico sin anclar el ciclo.  
  - Si no hay ningún `0`, la API responde **400** indicando que faltan datos de secado.

- **`previous_points`**  
  - Debe haber **al menos** tantas filas en ese tramo recortado como `previous_points`; si no, **400**.  
  - Los valores `previous_values` son los primeros `previous_points` valores de `soil_moisture_40`: son la **entrada acumulada** que el modelo usa en `predict_steps(...)`.  
  - `current_date` es la fecha del **último** de esos puntos (`iloc[previous_points - 1]`); `current_step` se fija a `previous_points`. Así el modelo sabe **hora/calendario** y **paso** a partir del código original del algoritmo.

- **`predict_steps`**  
  - Se pasa directamente a `model.predict_steps(..., future_steps=request.predict_steps)` y define la **longitud del vector de predicciones** futuras.

- **`model`**  
  - **`"ML"`**: se instancia `MLModel()`, se llama a `train(df)` con el `df` ya preparado (entrenamiento en cada petición, sin pesos persistentes en disco para ese modo).  
  - **`"Linear"`**: hace falta **`sensor_id`** no vacío; se carga `models/weights/sensor_{sensor_id}.joblib` (con caché en memoria). Si el archivo no existe → **503** indicando que hay que entrenar antes con `POST /train`.

- **`sensor_id`**  
  - Solo **obligatorio para Linear**, porque determina **qué pesos** se cargan; el diseño es **un archivo de pesos por sensor** (véase `docs/pregunta.md`).

Tras las predicciones, el detector de capacitancias recibe el mismo `df` (serie ya alineada al ciclo) para marcar eventos en el tiempo.

---

## 4. Qué datos devolvemos (respuesta)

El esquema de salida es `PredictResponse` en `api.py`:

| Campo | Significado |
|--------|-------------|
| `previous_values` | Lista de floats: los mismos `previous_points` valores de humedad usados como entrada (útiles para dibujar “lo observado” antes de la predicción). |
| `predictions` | Lista de floats: **un valor por paso futuro** (longitud = `predict_steps`, salvo detalles internos del modelo que ya devuelven lista acotada). |
| `prediction_dates` | Lista de strings: **misma longitud** que `predictions`. Instantánea de cada valor predicho: última fecha de `dates` + 30 min × (índice + 1) entre elementos consecutivos (alineado con `predict_steps` en Linear/ML). |
| `capacitances` | Lista de `{ "date", "value" }`: fechas detectadas como capacitancia y valor asociado (`capacitancy` del detector), para marcas verticales en gráficos. Si el detector falla, puede quedar vacía. |
| `dates` | Lista de strings: fechas de los **mismos** `previous_points` puntos (alineadas con `previous_values`). |
| `ccpmp` | Float: **mínimo** de las predicciones (o el último `previous_values` si no hay predicciones); umbral de humedad “más bajo esperado” en el horizonte predicho. |

Ejemplo ilustrativo de forma de respuesta:

```json
{
  "previous_values": [0.32, 0.31, 0.30],
  "predictions": [0.29, 0.28, 0.27],
  "prediction_dates": ["2024-06-07 08:30:00", "2024-06-07 09:00:00", "2024-06-07 09:30:00"],
  "capacitances": [{"date": "2024-06-08T10:00:00", "value": 0.25}],
  "dates": ["2024-06-06 12:00:00", "2024-06-06 12:30:00", "2024-06-07 08:00:00"],
  "ccpmp": 0.27
}
```

*(Los números son ficticios; la longitud real de listas depende de `previous_points` y `predict_steps`.)*

---

## 5. Errores HTTP que debe conocer el frontend / backend

- **400**: `data` vacío; sin ningún `irrigation_volume_0 == 0`; menos puntos disponibles que `previous_points`; `model` inválido; `Linear` sin `sensor_id`.  
- **503**: modelo `Linear` elegido pero **no existen pesos** para ese `sensor_id`.  
- Mensajes en `detail` según FastAPI / HTTPException.

---

## 6. Relación con el “creador del algoritmo”

- Los **modelos** (`predict_steps`, features de hora/estación, etc.) son los implementados en `models/ml_model.py` y `models/linear_model.py`; la API **no cambia la matemática**, solo **empaqueta entradas/salidas** y fija reglas de negocio (corte por riego, mínimos de puntos, rutas de pesos).  
- Cualquier ajuste fino (p. ej. filtrado de mesetas para el modelo lineal, umbrales de gradiente) que el autor documentó en comentarios de `main.py` sigue siendo relevante para **cómo se construye la serie en el cliente** antes de enviarla; la API asume que el JSON ya representa una ventana coherente con al menos un tramo post-riego con `irrigation_volume_0 == 0`.

---

## 7. Referencias rápidas en el repo

- Implementación HTTP: `api.py` (`PredictRequest`, `PredictResponse`, endpoint `POST /predict`).  
- Flujo Symfony / Elasticsearch / frontend: `docs/integracion.md`.  
- Preguntas de validación con el diseñador: `docs/pregunta.md`.  
- Uso en consola del mismo pipeline conceptual: `main.py`.
