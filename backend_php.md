# Informe: predicciones SAF ML con `iaForecast` en la normalización

## Alcance

Describe el proceso cuando un dispositivo tiene activado el flag **`iaForecast`** en base de datos y se ejecuta la normalización de datos originales (comando de consola habitual del flujo). Solo aplica a variables de humedad admitidas por el pipeline: **`soil_moisture`** y **`available_water`**. El resto de variables no dispara llamadas al servicio de predicción.

## Cuándo y cuántas veces se predice

La predicción **no** se hace fila a fila. Durante el batch, cada lectura normalizada válida sigue el camino habitual hacia el bulk de Elasticsearch.

### Qué es una «clave» (serie de pronóstico)

El backend agrupa las lecturas de humedad en **series independientes**. Una serie se identifica de forma inequívoca por **tres datos tomados del propio registro**:

1. **`sensor.deviceSensorid`** — identificador del sensor tal como viaja en el dato (no el id interno de base de datos del sensor si difiere).
2. **`position`** — posición numérica del punto de medida en el dispositivo.
3. **`variable.name`** — solo cuenta si es `soil_moisture` o `available_water`; son **dos series distintas** aunque sea el mismo sensor y la misma posición.

Cada combinación distinta de esos tres valores es una **clave** distinta. Un mismo dispositivo físico puede tener **varias claves a la vez** (varios sensores, varias profundidades/posiciones, o humedad de suelo frente a agua disponible).

### Una predicción por clave y por lote

Si el dispositivo tiene previsiones activas, el sistema **acumula** en memoria todas las filas de humedad del batch y, **por cada clave**, va guardando solo el candidato con la **fecha más reciente** de ese lote como **ancla** de esa serie.

**Cuando termina de recorrer todo el lote** de ese dispositivo, ejecuta el flujo de forecast **tantas veces como claves tengan candidato**: una llamada completa (pesos + predict + purga + inserción) **por clave**, siempre usando el ancla más nueva de esa clave en ese batch. No hay una segunda predicción para lecturas intermedias de la misma clave en el mismo lote; esas filas solo sirven para enriquecer el histórico en memoria y poder formar los dos puntos previos al ancla.

**Ejemplo:** un dispositivo con dos sensores de suelo en la misma posición 1 pero `deviceSensorid` A y B genera **dos claves** para `soil_moisture` → hasta **dos** predicciones al cerrar el lote. Si además hay `available_water` para el sensor A en posición 1, es una **tercera** clave → una tercera predicción.

## Requisitos para que llegue a llamarse al servidor

1. **`iaForecast`** verdadero en el dispositivo.
2. Variable **`soil_moisture`** o **`available_water`**.
3. Dato **no marcado como skip** en la normalización.
4. Variable de entorno **`SAF_ML_API_URL`** definida (API SAF ML alcanzable).
5. **Al menos dos lecturas anteriores** al instante ancla: se obtienen mezclando lecturas ya indexadas en Elasticsearch (solo documentos **reales**, `forecast` falso) con puntos del **mismo batch** aún no persistidos. Si no se alcanzan dos puntos previos, **no** se llama a la API.
6. **Modelo ML entrenado**: primero se hace una petición **GET** a la ruta de **pesos** del sensor (`…/sensors/{sensor_id}/weights`). Debe indicar que el bloque **ML** tiene pesos (`has_weights`). No se usa el modelo lineal para esta predicción; si solo hubiera lineal, el flujo se detiene aquí.

## Qué se envía al servidor (orden real)

**1) Comprobación de pesos (GET)**  
URL base configurada + `/sensors/{sensor_id}/weights`, cabecera Accept JSON. Respuesta distinta de éxito o ausencia de pesos ML ⇒ fin del flujo de predicción.

**2) Predicción (POST)**  
URL base + `/predict`, cuerpo JSON con esta forma lógica:

- **`sensor_id`**: identificador del sensor en la API (el mismo que en la comprobación de pesos).
- **`model`**: siempre la cadena **`ML`** cuando se acepta el flujo (solo si había pesos ML).
- **`data`**: lista de **tres** objetos, cada uno con:
  - **`date`**: instante en formato internacional compatible con el backend (p. ej. `2026-04-21 14:30:00`).
  - **`soil_moisture_40`**: valor numérico de humedad en la **escala que espera la API SAF** (en la práctica coincide con el valor normalizado usado en el pipeline; el riego se envía fijo a cero).
  - **`irrigation_volume_0`**: **0.0** en todos los puntos enviados por este integrador.
  Los dos primeros elementos son los **históricos** inmediatamente anteriores al ancla; el tercero es la **observación recién normalizada** (misma fecha que el dato ancla).

- **`previous_points`**: **3** (coherente con los tres puntos de la ventana).
- **`predict_steps`**: **100** (hasta cien valores futuros).

**Ejemplo ilustrativo del cuerpo de `POST /predict`**

```json
{
  "sensor_id": "12345",
  "model": "ML",
  "data": [
    {
      "date": "2026-04-21 08:00:00",
      "soil_moisture_40": 32.1,
      "irrigation_volume_0": 0.0
    },
    {
      "date": "2026-04-21 12:00:00",
      "soil_moisture_40": 31.8,
      "irrigation_volume_0": 0.0
    },
    {
      "date": "2026-04-21 14:30:00",
      "soil_moisture_40": 31.5,
      "irrigation_volume_0": 0.0
    }
  ],
  "previous_points": 3,
  "predict_steps": 100
}
```

## Qué se espera que devuelva la API

Respuesta HTTP correcta con JSON que incluya al menos:

- **`predictions`**: lista de números (valores pronosticados de humedad en la misma escala que la entrada).
- **`prediction_dates`**: lista de cadenas de fecha/h correspondientes a cada predicción.

Ambas listas deben ser **arrays**; el backend recorta al mínimo entre cien pasos y la longitud de cada lista. Si falta alguno de los dos campos o la llamada falla, **no** se escribe pronóstico nuevo (la normalización del dato real ya se había encolado igualmente).

**Ejemplo ilustrativo de respuesta**

```json
{
  "predictions": [31.2, 31.0, 30.8, 30.6],
  "prediction_dates": [
    "2026-04-21 15:00:00",
    "2026-04-21 16:00:00",
    "2026-04-21 17:00:00",
    "2026-04-21 18:00:00"
  ]
}
```

(En producción la lista suele tener hasta 100 elementos; el ejemplo está acortado.)

## Qué se hace en Elasticsearch antes de guardar

Si la respuesta es válida y se generan documentos, se ejecuta un **borrado por consulta** en el **mismo índice** que usa ese dato en la normalización: elimina **todos** los documentos con **`forecast: true`** que coincidan con el dispositivo, el identificador de sensor en dispositivo, la posición y el nombre de variable. No filtra por rango de fechas: sustituye el pronóstico previo de esa serie al completo. Si el borrado falla, se registra el error pero **se intenta igualmente** insertar el nuevo lote de pronósticos.

## Qué se acaba guardando en Elasticsearch

Hasta **100** documentos nuevos, uno por paso temporal devuelto. Cada documento:

- Reutiliza la **misma estructura** que una lectura de dispositivo normal (cliente, finca, dispositivo, sensor, variable, geolocalización, posición, profundidad, etc.) tomada del **dato ancla** real.
- Lleva **`forecast: true`** y **`updated: true`**.
- **`id`** único: identificador del dato normalizado base + sufijo fijo por índice de paso (patrón `…_saf_f_0`, `…_saf_f_1`, …).
- **`date`**: fecha del **paso pronosticado** (derivada de `prediction_dates`).
- **`creation_date`**: instante de creación del documento de pronóstico (tiempo actual en inserción).
- En **`variable`**: `normalized_value` y `original_value` coinciden con el **valor predicho** en texto.
- Bajo la clave con el **nombre de la variable** (`soil_moisture` o `available_water`), objeto **`value`** con la unidad por defecto de la variable y el **número predicho**.

La escritura es un **bulk** de indexación con esos `_id`, en el índice de actualización asociado al flujo de normalización de ese registro.

**Ejemplo ilustrativo de un documento almacenado** (campos representativos; valores ficticios)

```json
{
  "id": "abc123_saf_f_0",
  "date": "2026-04-21T15:00:00+02:00",
  "creation_date": "2026-04-21T14:35:12Z",
  "forecast": true,
  "updated": true,
  "deleted": false,
  "position": 1,
  "geolocation": "…",
  "depth": "…",
  "client": { "id": 10, "name": "…" },
  "farm": { "id": 20, "name": "…" },
  "device": { "id": 100, "name": "…", "alias": "…" },
  "sensor": {
    "id": 200,
    "deviceSensorid": "12345",
    "idDecagon": "…",
    "name": "…",
    "port": 1
  },
  "variable": {
    "name": "soil_moisture",
    "default_value_name": "cb",
    "original_value": "31.2",
    "normalized_value": "31.2"
  },
  "soil_moisture": {
    "value": {
      "cb": 31.2
    }
  }
}
```

(La unidad `cb` es un ejemplo; la real es la **`default_value_name`** de la variable en el dato.)

## Comportamiento ante fallos

Cualquier incumplimiento de las guardas anteriores, error de red, HTTP no exitoso en pesos o en predict, o respuesta incompleta hace que el pronóstico **se omita** sin interrumpir la normalización: el dato observacional sigue su curso. Los fallos se dejan constancia en logs de depuración o de excepciones según el caso.

## Resumen operativo

| Etapa | Acción |
|--------|--------|
| Disparador | Fin de batch con `iaForecast` activo: una pasada de forecast por cada clave `(deviceSensorid, position, variable)` que tenga ancla en el lote. |
| Lecturas previas | Dos puntos reales anteriores (ES + buffer del batch). |
| API | GET pesos ML → POST `/predict` con ventana de 3 puntos y 100 pasos. |
| Elasticsearch | Purga `forecast:true` de la serie → bulk de hasta 100 docs `forecast:true`. |

---

*Documentación técnica ampliada en el repositorio: `docs/dispositivos/ia-forecast-normalizacion-saf-ml.md`.*
