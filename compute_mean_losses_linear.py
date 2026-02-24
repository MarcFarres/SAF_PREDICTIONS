import pandas as pd
import numpy as np
from utils import preprocess, models
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor, LinearRegression
import xgboost as xgb

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'
    
def line_regression(points):

    points = np.array(points)

    x = np.array([i for i in range(len(points))])
    y = points

    x0, y0 = x[0], y[0]

    X = (x - x0).reshape(-1, 1)
    Y = y - y0

    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        min_samples=3,
        residual_threshold=1.5,  # tune this
        random_state=0
    )


    ransac.fit(X, Y)

    m = ransac.estimator_.coef_[0]

    return m

def get_plain_input(moisture, df, idx, steps):

    plain_input = [[moisture],
                   [steps],
                   df.loc[idx, ["season_autumn", "season_spring", "season_summer", "season_winter"]].values,
                   ]

    plain_input = np.concat(plain_input)


    plain_input = pd.DataFrame(plain_input.reshape(1, -1), columns=["soil_moisture_40", "steps_from_peak", "season_autumn", "season_spring", "season_summer", "season_winter"])
    plain_input = plain_input.astype(df.dtypes[plain_input.columns])
    plain_input["hour_s"] = np.sin(2 * np.pi * df.loc[idx, "date"].hour)
    plain_input["hour_c"] = np.cos(2 * np.pi * df.loc[idx, "date"].hour)
    return plain_input

def update_prediction(df_predicted, m, b, starting_idx):

    MAX_STEPS = 100
    
    dates = [df_predicted.loc[starting_idx, "date"]]
    values = [df_predicted.loc[starting_idx, "soil_moisture_40"]]

    plain_input = get_plain_input(df_predicted.loc[starting_idx, "soil_moisture_40"], df_predicted, starting_idx, 0)
    plain_steps = 0
    stop_steps = None
    
    for step in range(MAX_STEPS):

        if df_predicted.loc[starting_idx + step, "irrigation_volume_0"] > 0:
            stop_steps = step
            break

        if plain_model.predict(plain_input):
            plain_steps += 1
            prev_val = values[-1]
            values.append(prev_val)
        
        else:
            values.append(m*(step-plain_steps) + b)

        plain_input = get_plain_input(values[-1], df_predicted, starting_idx, step)


    if stop_steps is None:
        stop_steps = MAX_STEPS

    return values, df_predicted.loc[starting_idx:starting_idx + stop_steps, "soil_moisture_40"].values

df = pd.read_csv("data/1082-Device-Data-Fix.csv")
df = preprocess.get_clean_df(df)
df = preprocess.create_standarized_gradients(df)
df = preprocess.create_steps_from_irrigation(df)

THRESH_DOWN = -0.1
THRESH_UP = 0.1

plain_model = models.get_plain_predictor()

plain_dates = preprocess.get_plains_dates(df, THRESH_DOWN, THRESH_UP)
df["plain"] = False
df.loc[df["date"].isin(plain_dates), "plain"] = True
df_decay = preprocess.get_dataset_from_df(df, THRESH_UP)

X = preprocess.get_inputs_plain_predictor(df_decay)
df_decay["plain"] = plain_model.predict(X)
df_decay["plain"] = df_decay["plain"].astype(bool)

X.head()

start_date = pd.to_datetime("2024-06-06 12:00:00")
end_date   = pd.to_datetime("2024-06-30 05:00:00")
df_constricted = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
#We use the gradients to detect slopes and plains, but we first normalize to be able to get good scales for thresholding

restricted_mask = ~df_decay["plain"] & ~df_decay["up"]
restricted_dates = df_decay.loc[restricted_mask, "date"]

restricted_df = df_constricted.copy()
restricted_df["soil_moisture_40"] = np.nan
restricted_df["soil_moisture_40"] = df_constricted.loc[df_constricted["date"].isin(restricted_dates), "soil_moisture_40"]

df_decay_grouped = preprocess.separate_slopes_by_groups(df_decay)
df_decay_grouped = preprocess.compute_group_slopes(df_decay_grouped)
X, y = preprocess.compute_slopes_dataset(df_decay_grouped)
X.head()

from tqdm import tqdm

start_date = pd.to_datetime("2024-04-01 12:00:00")
end_date   = pd.to_datetime("2024-09-30 05:00:00")
df_constricted = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

df_predicted = df.copy()

df_predicted["season"] = df_predicted["date"].dt.month.apply(get_season)

df_predicted = pd.get_dummies(df_predicted, columns=['season'])

df_predicted = df_predicted[(df_predicted["date"].dt.year > end_date.year)]
mask = df_predicted["irrigation_volume_0"] > 0
first_point = mask[mask].index[0]
df_predicted = df_predicted.loc[first_point:].copy()

print(len(df_predicted))

epoch = 0


beta = 0.3

irrigation = False
current_points = []
current_m = None
current_b = None
beginning_idx = df_predicted.index[0]
final_predicted = []
new_prediction = None
error_rows = []
# inside loop:

for idx in tqdm(df_predicted.index, desc="Processing rows:", unit="row"):

    if df_predicted.loc[idx, "irrigation_volume_0"] > 0 or df_predicted.loc[idx, "gradient_std"] > 0.4:
        current_m = None

        if new_prediction is not None:
            final_predicted.append(new_prediction.loc[beginning_idx:idx])

        new_prediction = None
        beginning_idx = idx
        current_points = []
        
        continue


    if not (df_predicted.loc[idx, "gradient_std"] > THRESH_DOWN) & (df_predicted.loc[idx, "gradient_std"] <= THRESH_DOWN):
        current_points.append(df_predicted.loc[idx, "soil_moisture_40"])

    
    if len(current_points) > 3:
        m1 = line_regression(current_points)
        current_m = beta * m1 + (1-beta)*current_m
        current_b = current_points[-1]

    elif len(current_points) == 3:
        current_m = line_regression(current_points)
        current_b = current_points[0]

    if current_m != None:
        epoch += 1
        try:
            
            predicted, real_values = update_prediction(df_predicted, current_m, current_b, idx)
            row = np.abs(predicted - real_values)
            error_rows.append(row)
            
        except:
            break
        
        
        if epoch%5000 == 0:
            print("Saving errors just in case")
            errors = pd.DataFrame(error_rows)

            mae_per_step = errors.mean().to_numpy()
            np.save("results/errors_linear.npy", mae_per_step)

        # create_frame(new_prediction.loc[beginning_idx:], df_constricted,epoch,final_predicted)
        # create_frame_pointed(df_predicted, new_prediction, epoch, idx)

print("Saving errors")
errors = pd.DataFrame(error_rows)
mae_per_step = errors.mean().to_numpy()
np.save("results/errors_linear.npy", mae_per_step)
