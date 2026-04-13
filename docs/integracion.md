# Integración SAF ↔ Symfony (PHP)

Guía para consumir la **API REST FastAPI** del proyecto SAF desde un backend Symfony: endpoints disponibles, cuerpos JSON, respuestas, códigos de error y puntos prácticos (HTTP client, Elasticsearch, despliegue).

## Visión general

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Symfony    │────▶│  API Python │────▶│  modelos    │
│             │     │  (HTTP)      │     │  (FastAPI)  │     │  .joblib    │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
       ▲                     ▲                     │
       │                     │                     ▼
       │                     │            models/weights/*.joblib
       │              ┌──────┴──────┐
       └──────────────│ JSON resp. │
                      └─────────────┘
```

- Symfony actúa como **puerta de enlace**: valida sesión, monta payloads y opcionalmente persiste resultados.
- La API Python **no entrena en cada predicción** para ML: exige pesos por sensor en disco (salvo que primero llames a `POST /train`).
- Los datos de serie temporal deben llevar **fecha**, **humedad a 40 cm** y **volumen de riego** en el mismo formato que define la API.

**Documentación interactiva:** al tener levantado el servicio, `GET /docs` (Swagger UI) y `GET /openapi.json`.

---

## Formato común: `DataPoint`

Tanto `/train` como `/predict` usan un array de objetos con la misma forma:

| Campo | Tipo | Obligatorio | Notas |
|--------|------|-------------|--------|
| `date` | string | sí | Parseable por pandas (ej. `2024-06-06 12:00:00`). |
| `soil_moisture_40` | number | sí | Humedad normalizada a 40 cm (convención del proyecto). |
| `irrigation_volume_0` | number | no | Por defecto `0`. Debe existir **al menos un punto con valor `0`** en train y en predict (tramo post-riego / secado). |

Symfony puede mapear documentos de Elasticsearch (u otra fuente) a estos tres campos antes de serializar a JSON.

---

## Tabla de endpoints

| Método | Ruta | Uso |
|--------|------|-----|
| `GET` | `/health` | Comprobar que el servicio responde. |
| `GET` | `/sensors/{sensor_id}/weights` | Saber si existen pesos **Linear** y **ML** para ese sensor (rutas en el servidor de la API). |
| `POST` | `/train` | Entrenar **Linear** o **ML** con histórico largo y guardar `.joblib` en la máquina donde corre la API. |
| `POST` | `/predict` | Predecir usando ventana reciente; **Linear** y **ML** cargan pesos del disco para el `sensor_id` indicado. |

**Cabeceras recomendadas**

- `Content-Type: application/json; charset=utf-8` en todos los `POST` con cuerpo JSON (Symfony `HttpClient` con opción `json` lo envía correctamente).
- Si expones la API detrás de **ngrok** gratuito, suele ayudar `ngrok-skip-browser-warning: true` en peticiones automatizadas.

---

## `GET /health`

**Respuesta 200**

```json
{ "status": "ok" }
```

---

## `GET /sensors/{sensor_id}/weights`

Indica si hay archivos de pesos en el **filesystem del proceso** que ejecuta la API (rutas relativas típicas al directorio de trabajo).

**Respuesta 200 (ejemplo)**

```json
{
  "sensor_id": "1082",
  "linear": {
    "has_weights": true,
    "path": "models/weights/sensor_1082.joblib"
  },
  "ml": {
    "has_weights": true,
    "latest_path": "models/weights/ml_sensor_1082_20260211_143052.joblib",
    "all_paths": ["models/weights/ml_sensor_1082_20260211_143052.joblib"]
  }
}
```

- **Linear:** un solo fichero por sensor, `sensor_{sensor_id}.joblib`.
- **ML:** cero o más ficheros `ml_sensor_{sensor_id}_{YYYYMMDD}_{HHMMSS}.joblib` (marca de tiempo **UTC**). En predicción se usa el **más reciente** (orden por sufijo en el nombre).

Uso típico en Symfony: antes de ofrecer “Predecir con ML”, consultar este endpoint o manejar el **503** de `/predict` y mostrar al usuario que debe entrenar.

La convención de rutas de pesos está descrita también en el README del repositorio.

---

## `POST /train`

Entrena un modelo y **persiste** los pesos bajo `models/weights/` en el servidor de la API.

### Cuerpo (JSON)

| Campo | Tipo | Descripción |
|--------|------|-------------|
| `sensor_id` | string | Identificador del sensor (mismo que usarás en `/predict`). |
| `model` | string | `"Linear"` o `"ML"`. |
| `data` | `DataPoint[]` | Histórico; **mínimo 500 puntos** para Linear y para ML. |

**Reglas de negocio**

- `data` no puede estar vacío.
- Debe haber **al menos un** `irrigation_volume_0 == 0` (ciclos riego–secado).
- Linear y ML comparten el mismo mínimo de puntos (**500**) en la implementación actual.

### Respuesta 200 (ejemplo Linear)

```json
{
  "status": "ok",
  "sensor_id": "1082",
  "model": "Linear",
  "path": "models/weights/sensor_1082.joblib",
  "message": "Modelo Linear guardado en models/weights/sensor_1082.joblib"
}
```

### Respuesta 200 (ejemplo ML)

```json
{
  "status": "ok",
  "sensor_id": "1082",
  "model": "ML",
  "path": "models/weights/ml_sensor_1082_20260326_120530.joblib",
  "message": "Modelo ML guardado en models/weights/ml_sensor_1082_20260326_120530.joblib"
}
```

Cada entrenamiento ML **crea un archivo nuevo** con fecha en el nombre; los anteriores pueden quedar listados en `GET .../weights` → `ml.all_paths`.

### Errores frecuentes

- **400:** `model` inválido, pocos puntos, sin ningún `irrigation_volume_0 == 0`, datos incompatibles con el entrenamiento interno.
- **500:** fallo no controlado durante el entrenamiento.

**Rendimiento:** el entrenamiento puede tardar varios segundos o más; en Symfony conviene **timeout alto**, cola (Messenger) o job en segundo plano.

---

## `POST /predict`

Devuelve predicción de humedad, fechas alineadas, capacitancias detectadas y `ccpmp`.

### Cuerpo (JSON)

| Campo | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `data` | `DataPoint[]` | — | Obligatorio; no vacío. |
| `model` | string | `"ML"` | `"ML"` o `"Linear"`. |
| `sensor_id` | string | `""` | **Obligatorio** (no vacío) para ML y Linear. |
| `previous_points` | int | `10` | Ventana de contexto; **mínimo 3**. |
| `predict_steps` | int | `100` | Número de pasos futuros; **mínimo 1**; la longitud de `predictions` coincide normalmente con este valor. |

**Procesamiento en la API (resumen)**

1. Ordena `data` por `date`.
2. Recorta desde el **primer** punto con `irrigation_volume_0 == 0` (inicio del tramo útil).
3. Exige al menos `previous_points` filas en ese tramo.
4. Carga el modelo desde disco (**503** si faltan pesos).
5. Ejecuta predicción y detector de capacidades sobre el mismo `df` recortado.

Por tanto, las fechas que el cliente envió **antes** del primer `irrigation_volume_0 == 0` **no** aparecen en `dates` ni en `previous_values`.

### Respuesta 200

| Campo | Tipo | Significado |
|--------|------|-------------|
| `previous_values` | `float[]` | Humedades de contexto (primeros `previous_points` del tramo recortado). |
| `predictions` | `float[]` | Valores predichos; longitud = `predict_steps` salvo comportamiento interno del modelo. |
| `prediction_dates` | `string[]` | **Misma longitud** que `predictions`. Instantánea de cada predicho: ancla en la última fecha de contexto + **30 min** × (índice + 1). |
| `capacitances` | `{ date, value }[]` | Eventos de capacidad; puede ser `[]`. |
| `dates` | `string[]` | Fechas alineadas con `previous_values` (misma longitud). |
| `ccpmp` | float | Mínimo de `predictions` si hay elementos; si no, último `previous_values`. |

**Ejemplo (forma ilustrativa)**

```json
{
  "previous_values": [0.32, 0.31, 0.30],
  "predictions": [0.29, 0.288, "..."],
  "prediction_dates": ["2024-06-07 09:00:00", "2024-06-07 09:30:00", "..."],
  "capacitances": [{ "date": "2024-06-08T10:00:00", "value": 0.25 }],
  "dates": ["2024-06-07 07:00:00", "2024-06-07 07:30:00", "2024-06-07 08:00:00"],
  "ccpmp": 0.27
}
```

### Errores frecuentes

- **400:** `data` vacío; sin ningún `irrigation_volume_0 == 0`; menos filas tras el corte que `previous_points`; `model` no es `ML` ni `Linear`; `sensor_id` vacío con ML o Linear.
- **503:** no existe fichero Linear `models/weights/sensor_{id}.joblib` o no hay ningún ML `models/weights/ml_sensor_{id}_*.joblib` para ese sensor. El `detail` sugiere llamar a `POST /train` con el `model` adecuado y **mínimo 500** puntos en el histórico de entrenamiento.

Los errores de validación de FastAPI/Pydantic suelen devolver **422** con un JSON `detail` estructurado; los mensajes de negocio anteriores usan **400** o **503** con `detail` string o lista según el caso.

---

## Integración Symfony (HttpClient)

Parámetro de configuración sugerido: `SAF_API_URL` (sin barra final), por ejemplo `http://saf-api:8000` o la URL pública si usas un reverse proxy.

### Ejemplo: predicción

```php
// src/Service/SafApiClient.php
use Symfony\Contracts\HttpClient\HttpClientInterface;

class SafApiClient
{
    public function __construct(
        private HttpClientInterface $httpClient,
        private string $safApiUrl,
    ) {}

    public function predict(string $sensorId, array $dataPoints, string $model = 'ML', int $previousPoints = 10, int $predictSteps = 100): array
    {
        $response = $this->httpClient->request('POST', $this->safApiUrl . '/predict', [
            'json' => [
                'sensor_id' => $sensorId,
                'model' => $model,
                'data' => $dataPoints,
                'previous_points' => $previousPoints,
                'predict_steps' => $predictSteps,
            ],
            'timeout' => 120,
            'headers' => [
                'ngrok-skip-browser-warning' => 'true', // solo si aplica
            ],
        ]);

        return $response->toArray();
    }

    public function getWeights(string $sensorId): array
    {
        $response = $this->httpClient->request('GET', $this->safApiUrl . '/sensors/' . rawurlencode($sensorId) . '/weights', [
            'timeout' => 30,
        ]);

        return $response->toArray();
    }

    public function train(string $sensorId, string $model, array $dataPoints): array
    {
        $response = $this->httpClient->request('POST', $this->safApiUrl . '/train', [
            'json' => [
                'sensor_id' => $sensorId,
                'model' => $model,
                'data' => $dataPoints,
            ],
            'timeout' => 600,
        ]);

        return $response->toArray();
    }
}
```

La opción **`json`** del cliente de Symfony serializa el array y envía **`Content-Type: application/json`**, evitando el error en el que el cuerpo llega como texto plano y FastAPI no parsea el objeto.

Tras `request()`, comprobar `$response->getStatusCode()`; en **4xx/5xx** usar `$response->getContent(false)` o el formato de excepciones de tu capa HTTP para devolver `detail` al frontend.

---

## Mapeo Elasticsearch → `DataPoint`

El mapeo exacto depende del índice. Objetivo: un array ordenado cronológicamente con `date`, `soil_moisture_40`, `irrigation_volume_0`.

```php
$data = [];
foreach ($elasticHits as $hit) {
    $source = $hit['_source'];
    $data[] = [
        'date' => $source['timestamp'], // formato compatible con la API
        'soil_moisture_40' => (float) ($source['moisture_40cm'] ?? $source['soil_moisture']),
        'irrigation_volume_0' => (float) ($source['irrigation_volume'] ?? 0),
    ];
}
```

Para **entrenar** suele hacer falta un rango largo (≥ 500 puntos). Para **predecir**, basta una ventana reciente coherente con al menos un `irrigation_volume_0 == 0` y suficientes puntos tras el corte para `previous_points`.

---

## Flujo recomendado en producto

1. **Primera vez / nuevo sensor:** backoffice o proceso batch obtiene histórico ≥ 500 puntos y llama `POST /train` con `model: "ML"` y/o `"Linear"`.
2. **Antes de predecir (opcional):** ` GET /sensors/{id}/weights` para mostrar en UI si hay modelo disponible.
3. **Predicción en tiempo casi real:** Symfony obtiene los últimos puntos del sensor, llama `POST /predict` con el mismo `sensor_id` y el `model` elegido.
4. El frontend grafica `dates` + `previous_values`, y `prediction_dates` + `predictions`, más marcas de `capacitances` si aplica.

Si `/predict` puede superar el timeout del PHP-FPM, usar **Messenger + worker** que llame a la API y guarde el resultado para que el frontend haga polling.

---

## Despliegue y seguridad

- Ejecutar la API con `uvicorn api:app --host 0.0.0.0 --port 8000` (o detrás de un reverse proxy). Los `.joblib` deben residir en un volumen persistente en el host del servicio SAF.
- Restringir acceso a la API (red interna, VPN, API key en cabecera validada por middleware propio o por el proxy).
- En **Docker**, montar `/app/models/weights` (o la ruta equivalente) como volumen para no perder entrenamientos al recrear el contenedor.

---

## Resumen de tareas Symfony

1. Configurar `SAF_API_URL` y un servicio HTTP que encapsule `GET /health`, `GET /sensors/{id}/weights`, `POST /train` y `POST /predict`.
2. Implementar el mapeo desde Elasticsearch (u otra fuente) al formato `DataPoint`.
3. Exponer rutas de aplicación que el frontend use para entrenar (si procede) y predecir, manejando **503** (falta entrenar) y timeouts en entrenamiento.
4. Opcional: persistir resultados de predicción y metadatos (`ccpmp`, modelo, fechas) para histórico y gráficos.

Si compartes el esquema real del índice Elasticsearch (nombres de campos de tiempo, humedad y riego), el bloque de mapeo PHP se puede sustituir por una versión cerrada y testeable.
