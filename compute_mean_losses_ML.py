import pandas as pd
import numpy as np
from utils import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.dates as mdates
from utils.models import *
from utils.preprocess import *
import xgboost as xgb
from utils.eval import *
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'
    
def get_model_input(moisture, df, idx, steps):

    model_input = [[moisture],
                   [steps],
                   df.loc[idx, ["season_autumn", "season_spring", "season_summer"]].values,
                   ]

    model_input = np.concat(model_input)


    model_input = pd.DataFrame(model_input.reshape(1, -1), columns=["Input_0", "current_step", "season_autumn", "season_spring", "season_summer"])
    model_input = model_input.astype(X_new.dtypes[model_input.columns])
    model_input["hour_s"] = np.sin(2 * np.pi * df.loc[idx, "date"].hour)
    model_input["hour_c"] = np.cos(2 * np.pi * df.loc[idx, "date"].hour)
    model_input = model_input[["Input_0", "current_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]]
    return model_input

def update_prediction(df_predicted, model, starting_idx):

    MAX_STEPS = 100
    
    dates = [df_predicted.loc[starting_idx, "date"]]
    values = [df_predicted.loc[starting_idx, "soil_moisture_40"]]
    stop_steps = None
    
    for step in range(0, MAX_STEPS, 3):

        if df_predicted.loc[starting_idx + step, "irrigation_volume_0"] > 0:
            stop_steps = step
            break

        model_input = get_model_input(values[-1], df_predicted, starting_idx, step)

        # dates = np.concat(dates, df_predicted.loc[starting_idx + step:starting_idx + step + 3, "date"])

        new_values = model.predict(model_input)
        
        values = np.concat([values, new_values.flatten()])

    if stop_steps is None:
        stop_steps = MAX_STEPS + 2

    df = pd.DataFrame({
    "date": df_predicted.loc[starting_idx:starting_idx + stop_steps, "date"],
    "soil_moisture_40": values,
    "real_values" : df_predicted.loc[starting_idx:starting_idx + stop_steps, "soil_moisture_40"]
    })

    return df

df = pd.read_csv("data/1082-Device-Data-Fix.csv")
df = preprocess.get_clean_df(df)

start_date = pd.to_datetime("2024-04-01")
end_date   = pd.to_datetime("2024-10-10")
df_constricted = df[(df["date"].dt.month >= start_date.month) & (df["date"].dt.month <= end_date.month) & (df["date"].dt.year <= end_date.year)].copy()
# df_constricted = df[(df["date"] >= start_date) & (df["date"] <= end_date) & (df["date"].dt.year <= end_date.year)].copy()
print(df_constricted.head())
orig_values = df_constricted["soil_moisture_40"].to_numpy()
values = gaussian_filter1d(orig_values, sigma=2)
df_constricted["soil_moisture_40"] = values

df_constricted['prev_value'] = df_constricted['soil_moisture_40'].shift(1)
df_constricted['difference'] = df_constricted['soil_moisture_40'] - df_constricted['prev_value']

df_constricted["peak"] = False
df_constricted["steps_from_peak"] = 0

GRAD_THRESH_HIGH = 0.001

current_steps = -1
up = False

for row in df_constricted.iloc[1:].itertuples(index=True):
    dx = abs(row.difference)
    down = row.difference <= 0
    current_steps += 1
    
    if not down:
        df_constricted.loc[row.Index, "peak"] = True
        current_steps = -1
    else:
        df_constricted.loc[row.Index, "steps_from_peak"] = current_steps


df_constricted["soil_moisture_40"] = orig_values

df_decay = df_constricted[~df_constricted["peak"]]
df_decay = df_decay.drop(columns=["irrigation_volume_0", "irrigation_volume_accumulated_0", "prev_value", "difference", "peak"])[1:].reset_index(drop=True)
df_decay["hour"] = df_decay["date"].dt.hour
df_decay["season"] = df_decay["date"].dt.month.apply(get_season)
df_decay = pd.get_dummies(df_decay, columns=['season'])
df_decay['soil_moisture_next'] = df_decay['soil_moisture_40'].shift(-1)
df_decay['next_steps'] = df_decay['steps_from_peak'].shift(-1)
df_decay = df_decay[df_decay["next_steps"] != 0]
df_decay = df_decay.drop(columns=["date", "soil_moisture_20", "soil_moisture_60", "next_steps"])[:-1]
df_decay.head()

X_train = df_decay[["soil_moisture_40", "steps_from_peak", "hour", "season_autumn", "season_spring", "season_summer"]].copy()
X_train["hour"] = X_train["hour"] / 24
print(X_train.head())

y_train = df_decay["soil_moisture_next"]
print(y_train)

ESTIMATORS = 700
DEPTH = 3
DIFF = True

X_new = X_train.copy()

X_new["Input_0"] = X_new["soil_moisture_40"]
X_new["Output_1"] = X_new["soil_moisture_40"].shift(-1)
X_new["Output_2"] = X_new["soil_moisture_40"].shift(-2)
X_new["Output_3"] = X_new["soil_moisture_40"].shift(-3)
X_new["season_autumn"] = X_new["season_autumn"].astype(bool)
X_new["season_summer"] = X_new["season_summer"].astype(bool)
X_new["season_spring"] = X_new["season_spring"].astype(bool)
X_new["current_step"] = X_new["steps_from_peak"].shift(-3)
X_new["hour"] = X_new["hour"]
X_new["hour_s"] = np.sin(2 * np.pi * X_new["hour"])
X_new["hour_c"] = np.cos(2 * np.pi * X_new["hour"])
condition = X_new["current_step"] >= 3
condition_full = ~X_new["Output_3"].isna() & condition
X_new = X_new.loc[condition_full]

y_new = X_new[["Output_1", "Output_2", "Output_3"]]
X_new = X_new[["Input_0", "current_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]]

xgb_reg = XGBRegressor()
xgb_reg.fit(X_new, y_new)
xgb_pred = xgb_reg.predict(X_new)
pd.DataFrame(xgb_pred, columns=['Y1', 'Y2', 'Y3'])
mae = mean_absolute_error(y_new, xgb_pred)
print("MAE:", mae)

from tqdm import tqdm
    
start_date = pd.to_datetime("2024-04-01 12:00:00")
end_date   = pd.to_datetime("2024-09-30 05:00:00")
df_constricted = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

df_predicted = df.copy()

df_predicted["season"] = df_predicted["date"].dt.month.apply(get_season)

df_predicted = pd.get_dummies(df_predicted, columns=['season'])

df_predicted = df_predicted[(df_predicted["date"].dt.month >= start_date.month) & (df_predicted["date"].dt.month <= end_date.month) & (df_predicted["date"].dt.year > end_date.year)]
mask = df_predicted["irrigation_volume_0"] > 0
first_point = mask[mask].index[0]
df_predicted = df_predicted.loc[first_point:].copy()

epoch = 0

beginning_idx = df_predicted.index[0]
current_points = []
error_rows = []
first_month_seen = False

for idx in tqdm(df_predicted.index, desc="Processing rows:", unit="row"):
    
    if df_predicted.loc[idx, "date"].month == start_date.month and not first_month_seen:
        print("Returned to first month")
        first_month_seen = True
        current_points = []

    if df_predicted.loc[idx, "irrigation_volume_0"] > 0:
        current_points = []
        
        continue


    epoch += 1
    
    current_points.append(df_predicted.loc[idx, "soil_moisture_40"])

    try:
        new_prediction = update_prediction(df_predicted, xgb_reg, idx)
    except Exception as e:
        print(f"End or error : {e}")
        break

    predicted = new_prediction["soil_moisture_40"].values
    real_values = new_prediction["real_values"].values
    row = {}
    i = 0

    for i in range(len(predicted)):
        row[str(i)] = abs(predicted[i] - real_values[i])
        if df_predicted.loc[idx + i, "irrigation_volume_0"] > 0:
            row[str(i)] = np.nan
            
    error_rows.append(row)
    
    if epoch%5000 == 0:
        print("Saving errors just in case")
        errors = pd.DataFrame(error_rows)

        mae_per_step = errors.mean().to_numpy()
        np.save("results/errors_ML_nowinter.npy", mae_per_step)
        
    if df_predicted.loc[idx, "date"].month == end_date.month and first_month_seen:
        print("Last month of year")
        first_month_seen = False
    
print("Saving errors")
errors = pd.DataFrame(error_rows)
mae_per_step = errors.mean().to_numpy()
np.save("results/errors_ML_nowinter.npy", mae_per_step)

            
