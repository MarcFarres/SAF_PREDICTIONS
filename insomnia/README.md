# Pruebas Insomnia (API SAF)

## Importar

1. Abre **Insomnia** → **Application** → **Preferences** → **Data** → **Import Data** (o arrastra el archivo).
2. Elige **`SAF_API.insomnia.json`** de esta carpeta.
3. Activa el entorno **ngrok** (esquina superior izquierda). La variable `base_url` apunta a tu túnel; cámbiala si ngrok te asigna otra URL.

## Contenido

| Petición | Descripción |
|----------|-------------|
| `01 GET /health` | Comprueba que el servicio responde a través de ngrok. |
| `02 GET /sensors/1082/weights` | Estado Linear + ML para el sensor de prueba. |
| `03 POST /train (ML)` | **~520 puntos** sintéticos; crea `ml_sensor_1082_*.joblib`. Tarda un poco. |
| `04 POST /train (Linear)` | Mismos datos; crea `sensor_1082.joblib`. |
| `05 POST /predict (ML)` | Requiere haber ejecutado antes el train ML del mismo `sensor_id`. |
| `06 POST /predict (Linear)` | Requiere train Linear previo. |
| `07 POST /predict (ML, 3 pts integración PHP)` | Misma forma que el backend PHP: **3** filas en `data`, `previous_points: 3`, `predict_steps: 100`, todo con `irrigation_volume_0: 0`. Edita `sensor_id` y humedades en `payloads/predict_ml_3pts_integration.json` (o en el body de la petición) y regenera el export si quieres versionarlo en git. |

Todas las peticiones llevan la cabecera `ngrok-skip-browser-warning: true` para reducir avisos en el túnel gratuito. Los **POST** incluyen además `Content-Type: application/json`; si al pegar el body usas el modo **Text** en lugar de **JSON**, FastAPI puede responder *Input should be a valid dictionary* porque el cuerpo llega como texto plano. En ese caso, cambia el body a tipo **JSON** o añade la cabecera `Content-Type: application/json`.

## Regenerar cuerpos JSON

Si cambias el sensor, la longitud de la serie o quieres otro patrón sintético:

```bash
cd SAF
python insomnia/generate_payloads.py
python insomnia/build_insomnia_export.py
```

Los archivos sueltos quedan en **`payloads/`** (por si prefieres pegar el body a mano). El export de Insomnia se reescribe con el contenido actualizado de esos ficheros.

## Orden recomendado

1. Arranca la API (`uvicorn api:app ...`) y ngrok.
2. `GET /health`
3. `GET /weights`
4. `POST /train` del modelo que vayas a usar (ML y/o Linear).
5. `POST /predict` correspondiente.

Si `/predict` responde **503**, el mensaje indica que falta entrenar o que no existen los `.joblib` para ese `sensor_id`.
