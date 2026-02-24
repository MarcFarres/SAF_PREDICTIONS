
# Estrategia de integración SAF ↔ Symfony PHP

## Visión general

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Symfony     │────▶│  API Python │────▶│  SAF Models │
│  (click)    │     │  (Controller) │     │  (FastAPI)  │     │  + Predict  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
       │                     │                     │                    │
       │                     │                     │                    │
       ▼                     ▼                     ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│ Chart.js /  │◀────│  JSON response│◀────│  predictions│◀────│  DataFrame │
│ ApexCharts  │     │  to frontend  │     │  capacitances│     │  processed  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ Elasticsearch│
                     │  (datos raw) │
                     └──────────────┘
```

---

## 1. API REST en Python (FastAPI)

Exponer SAF como microservicio que Symfony pueda consumir vía HTTP.

**Ventajas:** desacople, reutilizable, fácil de escalar y desplegar en Docker.

**Endpoints propuestos:**

| Endpoint | Método | Input | Output |
|----------|--------|-------|--------|
| `POST /predict` | POST | JSON con datos del sensor y opciones | predicciones + capacitancias |
| `GET /health` | GET | — | estado del servicio |

### Estructura del payload

```json
{
  "sensor_id": "1082",
  "model": "ML",
  "data": [
    {"date": "2024-06-06 12:00:00", "soil_moisture_40": 0.32, "irrigation_volume_0": 0},
    {"date": "2024-06-06 12:30:00", "soil_moisture_40": 0.30, "irrigation_volume_0": 0},
    ...
  ],
  "predict_steps": 100
}
```

PHP puede enviar exactamente los puntos que ya tiene de Elasticsearch (mapeados al formato anterior). La API Python se encarga del preprocesamiento, entrenamiento/carga del modelo, predicción y detección de capacitancias.

---

## 2. Servicio Symfony para llamar a la API

```php
// src/Service/SafPredictionService.php
class SafPredictionService
{
    public function __construct(
        private HttpClientInterface $httpClient,
        private string $safApiUrl  // config: SAF_API_URL
    ) {}

    public function predict(string $sensorId, array $elasticData, string $model = 'ML'): array
    {
        $payload = $this->transformElasticToSafFormat($elasticData);
        $payload['sensor_id'] = $sensorId;
        $payload['model'] = $model;

        $response = $this->httpClient->request('POST', $this->safApiUrl . '/predict', [
            'json' => $payload,
            'timeout' => 30,
        ]);

        return $response->toArray();
    }

    private function transformElasticToSafFormat(array $hits): array { /* ... */ }
}
```

El punto crítico es `transformElasticToSafFormat`: mapear cada documento de Elasticsearch a `date`, `soil_moisture_40` y `irrigation_volume_0`. Eso depende del índice y de tus campos reales.

---

## 3. Flujo por pasos

1. Usuario selecciona sensor y rango de fechas en la web.
2. Symfony consulta Elasticsearch y obtiene los documentos del sensor en ese rango.
3. Al hacer clic en “Predecir”, el frontend llama a una ruta Symfony (p. ej. `POST /api/sensor/{id}/predict`).
4. El controlador usa `SafPredictionService` para enviar esos datos a la API Python y recibe predicciones y capacitancias.
5. El controlador devuelve un JSON al frontend.
6. El frontend dibuja un gráfico (p. ej. Chart.js) con:
   - puntos reales
   - predicciones
   - líneas verticales para capacitancias

---

## 4. Flujo si la predicción tarda

Si el modelo tarda más de unos segundos, usar un flujo asíncrono:

```
Click → Symfony encola Job (Messenger) → Worker PHP llama API Python
       → Resultado en BD/Redis → Frontend hace polling o WebSocket
```

Se puede guardar el resultado en una entidad `PredictionResult` (sensor_id, created_at, predictions, capacitances, modelo) y el frontend consulta por ejemplo `GET /api/sensor/{id}/prediction/latest` hasta que termine.

---

## 5. Formato de datos Elasticsearch → SAF

En Elasticsearch sueles tener algo como:

- `timestamp`
- `moisture` o similar
- `irrigation` (volumen o flag)

Necesitas definir un mapeo concreto, por ejemplo:

```php
// Ejemplo de mapeo Elasticsearch → SAF
$data = [];
foreach ($elasticHits as $hit) {
    $source = $hit['_source'];
    $data[] = [
        'date' => $source['timestamp'],  // o el campo real
        'soil_moisture_40' => $source['moisture_40cm'] ?? $source['soil_moisture'],
        'irrigation_volume_0' => $source['irrigation_volume'] ?? 0,
    ];
}
```

Si el índice tiene otro esquema, el mapeo se ajusta, pero la idea es siempre producir un array de diccionarios con `date`, `soil_moisture_40` y `irrigation_volume_0`.

---

## 6. Despliegue de la API Python

**Opción A: Docker**

```dockerfile
# Dockerfile en el proyecto SAF
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Añadir: pip install fastapi uvicorn
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Opción B: mismo servidor**

Ejecutar `uvicorn` en un puerto (p. ej. 8000) y que Symfony llame a `http://localhost:8000` (o a la IP del servidor en producción).

---

## 7. Seguridad de la API

- Usar API key o token compartido: Symfony la envía en `Authorization` o header personalizado.
- O limitar el acceso a la red interna (firewall/VPN).
- En producción, exponer la API solo internamente, no al público.

---

## 8. Modelos preentrenados

Para producción conviene usar modelos ya entrenados y guardados:

- **Linear:** cargar `XGBoost_plain_classifier_new.joblib` en vez de entrenar en cada petición.
- **ML:** guardar el modelo tras el primer entrenamiento y cargarlo en cada arranque (o cachear una instancia en memoria en la API).

Así las respuestas son más rápidas y estables.

---

## 9. Resumen de tareas

1. Crear `api.py` (FastAPI) en el proyecto SAF con `POST /predict` y `GET /health`.
2. Implementar `transformElasticToSafFormat` en Symfony según el esquema real de Elasticsearch.
3. Crear `SafPredictionService` que llame a la API.
4. Añadir ruta y controlador en Symfony para `POST /api/sensor/{id}/predict`.
5. En el frontend, botón que llama a esa ruta y renderiza el gráfico con la respuesta.

Si compartes el esquema del índice de Elasticsearch (campos de fecha, humedad y riego), se puede afinar el mapeo y el contrato JSON de la API.