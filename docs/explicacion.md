
Aquí va un resumen claro del proyecto **SAF**:

---

## ¿Qué hace el proyecto SAF?

**SAF** es un sistema de **predicción y análisis de humedad del suelo** para riego agrícola. Usa datos de sensores (humedad en profundidad 40 cm) para:

1. Predecir la evolución futura de la humedad
2. Detectar momentos de “capacitancia” (mesetas donde la humedad se estabiliza)
3. Ayudar a decidir cuándo regar

---

## Estructura general

### 1. Datos de entrada

- CSV con medidas de humedad, normalmente con:
  - `date`: timestamp de la medida
  - `soil_moisture_40`: humedad del suelo a 40 cm
  - Información de riego (`irrigation_volume_0`)

- `preprocess_csv.py`: limpia CSVs que vienen con formato SAF (quita comillas y corrige el formato)
- `utils/preprocess.py`: prepara los datos (gradientes, estaciones, pasos desde el riego, etc.)

### 2. Modelos de predicción

Hay dos modelos para predecir la humedad futura:

#### LinearModel

- Regresión lineal (RANSAC) para modelar el descenso de humedad tras el riego
- Un clasificador XGBoost que indica si entramos en una **meseta** (plains)
- En mesetas la predicción se mantiene constante; en descenso se usa la pendiente lineal
- Usa estación del año y hora para mejorar la predicción
- Necesita filtrar bien los puntos “planos” para que la regresión sea correcta

#### MLModel

- Modelo basado en XGBRegressor
- Predice hasta 3 pasos por adelantado (90 minutos)
- Usa humedad actual, pasos desde el pico de riego, hora y estación
- Más simple de usar que el modelo lineal
- Los datos pasan por un filtro gaussiano antes del entrenamiento

### 3. CapacitanceDetector

- Localiza momentos de **capacitancia**, es decir, cuando la curva de humedad pasa de descenso a meseta
- Usa la segunda derivada de la humedad; cuando cambia de negativa a positiva indica ese cambio
- Sirve para detectar cuándo el suelo deja de drenar y se estabiliza
- El resultado son fechas donde aparece ese comportamiento

---

## Flujo típico (main.py)

1. Carga datos del sensor (ej. `data/1082-Device-Data-Fix.csv`)
2. Preprocesa con `preprocess.get_clean_df()`
3. Elige modelo: Linear o ML
4. Entrena (o carga) el modelo
5. A partir de los últimos valores de humedad, predice N pasos hacia adelante
6. Visualiza:
   - Predicciones vs valores reales
   - Humedad con líneas verticales en las capacitancias detectadas

---

## Tecnologías usadas

- **Pandas** y **NumPy** para datos
- **XGBoost** para modelos lineales/ML y detección de mesetas
- **Scikit-learn** para regresión, RANSAC y métricas
- **Optuna** para ajuste de hiperparámetros del detector de mesetas
- **Matplotlib** para gráficos
- **SciPy** para filtrado (gaussian_filter1d)

---

## Dependencias y entorno

```bash
pip install -r requirements.txt
```

Se espera un directorio `data/` con los CSVs ya preprocesados.

---

En conjunto, el proyecto implementa un **sistema de predicción de humedad del suelo** orientado a optimizar el riego usando sensores de humedad y modelos de machine learning.






Aquí tienes el flujo resumido paso a paso:

---

## Flujo completo de `main.py`

### Diagrama general

```
CSV crudo (SAF)                    CSV Fix                    DataFrame procesado
     │                                  │                              │
     │  preprocess_csv.py               │  pd.read_csv + get_clean_df   │
     └─────────────────────────────────┴──────────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────────┐
                              │  DataFrame listo   │
                              │  para los modelos  │
                              └────────────────────┘
```

---

## Paso a paso

### 1. Origen del CSV (antes de `main.py`)

El archivo `data/1082-Device-Data-Fix.csv` no se crea en `main.py`. Se obtiene antes con:

```bash
python preprocess_csv.py --path data/1082-Device-Data.csv
```

Eso genera `1082-Device-Data-Fix.csv` (fechas corregidas y comillas eliminadas). Este Fix.csv es el que carga `main.py`.

---

### 2. Carga y preprocesamiento (líneas 37–38)

```python
df = pd.read_csv("data/1082-Device-Data-Fix.csv")   # Carga el Fix
df = preprocess.get_clean_df(df)                      # Transforma a formato ancho
```

El CSV Fix está en **formato largo**: varias filas por fecha (una por variable/profundidad).  
`get_clean_df` hace:

- Eliminar columnas auxiliares (`sensor.deviceSensorid`, etc.)
- Crear `variable_label` = `variable.name` + `"_"` + `depth` (ej: `soil_moisture_40`)
- Convertir `date` a datetime
- Hacer un **pivot**: pasar de largo a ancho → una fila por fecha, columnas por variable

Resultado típico: columnas `date`, `soil_moisture_20`, `soil_moisture_40`, `soil_moisture_60`, `irrigation_volume_0`, `irrigation_volume_accumulated_0`.

---

### 3. Modelo (líneas 42–61)

Se crea un modelo (Linear o ML) y se entrena con `df` (ya transformado por `get_clean_df`).

---

### 4. Ventana de predicción (líneas 65–72)

```python
# Filtrar rango de fechas
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Buscar el primer punto donde NO hay riego (inicio del “decay”)
mask = df["irrigation_volume_0"] == 0
first_point = mask[mask].index[0]
df = df.loc[first_point:].copy()   # Solo datos desde ahí en adelante
```

Se usa solo una ventana de fechas y se elimina el tramo inicial con riego, dejando solo la fase de “secado”.

---

### 5. Predicción (líneas 74–143)

```python
previous_values = df["soil_moisture_40"].iloc[:10].values   # 10 valores “vistos”
current_date = df["date"].iloc[9]                            # Fecha del último valor
predictions = model.predict_steps(previous_values, current_date, 10, 100)
```

- Entrada: 10 valores de humedad y la fecha del último.
- Salida: predicción de 100 pasos futuros.

Para comparar se usan los valores reales:

```python
correct_values = df["soil_moisture_40"].iloc[10:110].values  # 100 valores reales
```

---

### 6. Resultados visibles

**Gráfica 1 – Predicción vs real**

- Eje X: “steps” (0–110)
- Puntos azules: los 10 valores “vistos”
- Línea roja: predicciones para los siguientes 100 pasos
- Línea verde: valores reales de esos mismos 100 pasos

Se usa para ver si el modelo sigue bien la evolución de la humedad.

**Gráfica 2 – Capacitancias**

- Eje X: fechas
- Línea: humedad `soil_moisture_40` a lo largo del tiempo
- Líneas verticales rojas: instantes de capacitancia (cambios de descenso a meseta)

Sirve para ver cuándo el suelo deja de drenar y se estabiliza.

---

## Dónde encaja cada parte

| Componente            | Función                          |
|-----------------------|----------------------------------|
| `preprocess_csv.py`   | Preparar CSV crudo → Fix.csv     |
| `preprocess.get_clean_df` | Pasar de largo a ancho y limpiar |
| Modelo (Linear / ML)  | Predecir evolución de humedad   |
| `CapacitanceDetector` | Marcar instantes de capacitancia |

---

## Flujo en una frase

`main.py` carga un CSV Fix (formato largo), lo convierte con `get_clean_df` a formato ancho, entrena un modelo de predicción de humedad y genera dos gráficas: una de predicción vs real y otra de humedad con capacitancias marcadas.