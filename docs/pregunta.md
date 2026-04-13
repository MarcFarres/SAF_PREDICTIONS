# Preguntas para validar la integración API con el creador del algoritmo

Documento resumen para validar con quien diseñó los modelos de predicción de humedad del suelo que la implementación de la API y el flujo de integración con el frontend son correctos.

---

## Resumen de la implementación

Se ha expuesto el proyecto SAF como API REST para que una aplicación web (Symfony PHP) pueda enviar datos y obtener predicciones de humedad sin ejecutar scripts manualmente.

---

## Puntos a validar

### 1. Dos modelos disponibles

Existen dos modelos de predicción:

- **ML:** Basado en XGBoost, entrena rápido con los datos que se le envían en cada petición.
- **Linear:** Combina regresión lineal (RANSAC) y un clasificador para detectar mesetas (plains).

¿Es correcto que haya estos dos modelos y que el usuario/frontend pueda elegir cuál usar?

---

### 2. Elección de modelo

El frontend puede indicar qué modelo usar en cada petición de predicción (por defecto ML).

¿El algoritmo está pensado para que se pueda alternar entre ambos según el caso de uso?

---

### 3. El modelo Linear requiere entrenamiento previo

El modelo Linear necesita un entrenamiento previo que puede durar varios minutos (búsqueda de hiperparámetros). Por eso:

- No se entrena en cada predicción.
- Primero se llama a un endpoint de entrenamiento con datos históricos del sensor.
- Los pesos se guardan y luego se reutilizan en las predicciones.

¿Es correcto que el Linear requiera este paso de entrenamiento explícito y separado de la predicción?

---

### 4. Un archivo de pesos por sensor

La decisión de diseño es: **cada sensor tiene su propio archivo de pesos**.

Motivación: un sensor en un suelo/clima/cultivo puede comportarse distinto a otro. Los pesos entrenados con datos del sensor 1082 no se aplican al sensor 3274.

Flujo:

1. Se entrena el modelo Linear con datos históricos del sensor X.
2. Se guarda un archivo de pesos para ese sensor (p. ej. `sensor_1082.joblib`).
3. Las predicciones del sensor X usan únicamente sus propios pesos.

¿Esta separación de pesos por sensor es la esperada desde el diseño del algoritmo?

---

### 5. Datos necesarios para entrenar

Para entrenar el modelo Linear se requieren:

- Mínimo unos 500 puntos históricos (semanas o meses de datos).
- Varios ciclos de riego–secado (puntos con riego y puntos sin riego).
- Para cada punto: fecha, humedad a 40 cm, volumen de riego.

¿Estos requisitos son razonables para que el entrenamiento sea fiable?

---

### 6. Datos necesarios para predecir

Para predecir (ambos modelos):

- Serie de puntos recientes con fecha, humedad y riego.
- Al menos un punto con riego = 0 (fase de secado).
- Mínimo unos 10 puntos previos para iniciar la predicción.

¿Coincide esto con lo que el algoritmo espera como entrada mínima?

---

## Endpoints disponibles

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Comprobar que la API responde |
| `GET /sensors/{id}/weights` | Saber si un sensor tiene pesos entrenados |
| `POST /train` | Entrenar el modelo Linear con datos históricos |
| `POST /predict` | Obtener predicciones y capacitancias |

---

## Síntesis para el creador del algoritmo

1. Hay dos modelos (ML y Linear) y se puede elegir entre ellos.
2. El Linear exige un paso previo de entrenamiento.
3. Cada sensor tiene su propio archivo de pesos; no se comparten entre sensores.
4. El frontend puede comprobar si un sensor está entrenado antes de permitir predicciones con Linear.
5. Los datos vienen de Elasticsearch (Symphony/PHP) y se transforman al formato que esperan los modelos.

Si alguno de estos puntos no coincide con el diseño original, sería importante ajustarlo antes de seguir con la integración.
