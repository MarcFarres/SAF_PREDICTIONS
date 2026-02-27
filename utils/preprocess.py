import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.ndimage import gaussian_filter1d

def get_clean_df(df : pd.DataFrame) -> pd.DataFrame:
    """
    Gets a clean and ordered DataFrame
    """
    
    df.drop(columns=["sensor.deviceSensorid", "position", "sensor.idDecagon", "variable.default_value_name"], inplace=True)
    
    mask = df["variable.name"].str.match("irrigation*")
    df.loc[mask, "depth"] = 0
    
    df["variable_label"] = df["variable.name"] + "_" + df["depth"].astype(int).astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%b %d %Y @ %H:%M:%S.%f")
    pivoted = df.pivot_table(index="date", columns="variable_label", values="variable.normalized_value", aggfunc="first").reset_index()
    pivoted.columns.name = None
    pivoted.sort_values(by="date", inplace=True)
    
    return pivoted

def merge_pred_with_prob(X, y, p=0.2):
    
    swap_mask = np.random.rand(len(X)) < p
    
    y_new = pd.DataFrame(data=y, columns=["soil_moisture_next"])
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"].shift(1)
    
    condition = X["steps_from_peak"] != 0
    
    swap_mask = swap_mask & condition
    
    X_res = X.copy()
    X_res.loc[swap_mask, "soil_moisture_40"] = y[swap_mask]
    
    print(f"Changed rows : {np.sum(swap_mask)}")
    
    return X_res

def merge_diff_with_prob(X, y, p=0.2):
    
    swap_mask = np.random.rand(len(X)) < p
    
    y_new = pd.DataFrame(data=y, columns=["soil_moisture_next"])
    
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"] + X["t2"]
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"].shift(1)
    
    condition = X["next_step"] >= 4
    
    swap_mask = swap_mask & condition
    
    X_res = X.copy()
    X_res.loc[swap_mask, "t2"] = y[swap_mask]
    
    print(f"Changed rows : {np.sum(swap_mask)}")
    
    return X_res

def format_XGBoost(input_vec):
    
    input_names = ["t0", "t1", "t2", "next_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]
    input_vec = pd.DataFrame(data=input_vec, columns=input_names)
    input_vec = xgb.DMatrix(input_vec)
    return input_vec

def format_GradientBoost(input_vec):
    input_names = ["soil_moisture_40", "steps_from_peak", "hour"]
    input_vec = pd.DataFrame(data=input_vec, columns=input_names)
    return input_vec
    
def get_window_data(kind: str, df: pd.DataFrame):
    """
    Gives the splits X and y for training with windows
    
    Parameters
    ----------
        kind : {'single', 'multi'}
            Selects wether you will do singular or multi-output
        df : `pd.DataFrame`
            The dataframe with the data already well structured

    Returns
    ----------
        X : `pd.DataFrame`
            A DataFrame with the input
        y : `pd.DataFrame`
            A DataFrame with the expected output
    """

    X_new = df.copy()

    if kind == "single":
        X_new["t0"] = df["soil_moisture_40"]
        X_new["t1"] = df["soil_moisture_40"].shift(-1)
        X_new["t2"] = df["soil_moisture_40"].shift(-2)
        X_new["res"] = df["soil_moisture_40"].shift(-3)
        X_new["season_autumn"] = df["season_autumn"].shift(-3).astype(bool)
        X_new["season_summer"] = df["season_summer"].shift(-3).astype(bool)
        X_new["season_spring"] = df["season_spring"].shift(-3).astype(bool)
        X_new["next_step"] = df["steps_from_peak"].shift(-3)
        X_new["hour"] = df["hour"].shift(-2)
        X_new["hour_s"] = np.sin(2 * np.pi * df["hour"])
        X_new["hour_c"] = np.cos(2 * np.pi * df["hour"])

        condition = X_new["next_step"] >= 3

        condition_full = ~X_new["t2"].isna() & condition

        X_new = X_new.loc[condition_full]

        y_new = X_new["res"] - X_new["t2"]
        X_new = X_new[["t0", "t1", "t2", "next_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]]

        return X_new, y_new
    
def create_standarized_gradients(df : pd.DataFrame) -> pd.DataFrame:
    orig_values = df["soil_moisture_40"].to_numpy()
    values = gaussian_filter1d(orig_values, sigma=1)
    df["soil_moisture_40"] = values

    df["gradient"] = df["soil_moisture_40"].diff()
    df["gradient_std"] = (df["gradient"] - df["gradient"].mean())/ df["gradient"].std()
    
    df["soil_moisture_40"] = orig_values

    return df

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def get_plains_dates(df : pd.DataFrame, threshold_down : float, threshold_up : float) -> pd.Series:
    filtered_gradients = (df["gradient_std"] > threshold_down) & (df["gradient_std"] <= threshold_up)
    filtered_dates = df.loc[filtered_gradients, "date"]

    return filtered_dates

def create_steps_from_irrigation(df : pd.DataFrame) -> pd.DataFrame:

    df["steps_from_peak"] = 0

    current_steps = -1

    for row in df.iloc[1:].itertuples(index=True):
        current_steps += 1

        if row.irrigation_volume_0 == 0:
            df.loc[row.Index, "steps_from_peak"] = current_steps
            
        else:
            current_steps = -1

    return df

def get_dataset_from_df(df : pd.DataFrame, threshold_up : float) -> pd.DataFrame:
    df_decay = df[df["irrigation_volume_0"] == 0].copy()
    df_decay["up"] = df_decay["gradient_std"] > threshold_up
    df_decay = df_decay.drop(columns=["irrigation_volume_0", "irrigation_volume_accumulated_0", "gradient", "gradient_std"])[1:].reset_index(drop=True)
    df_decay["hour"] = df_decay["date"].dt.hour
    df_decay["season"] = df_decay["date"].dt.month.apply(get_season)
    df_decay = pd.get_dummies(df_decay, columns=['season'])
    # Asegurar que existan las 4 columnas de estación (get_dummies solo crea las que hay en los datos)
    season_cols = ["season_autumn", "season_spring", "season_summer", "season_winter"]
    for col in season_cols:
        if col not in df_decay.columns:
            df_decay[col] = 0
    df_decay["hour_s"] = np.sin(2 * np.pi * df_decay["hour"])
    df_decay["hour_c"] = np.cos(2 * np.pi * df_decay["hour"])
    df_decay = df_decay.drop(columns=["soil_moisture_20", "soil_moisture_60", "hour"])[:-1]
    
    return df_decay

def get_inputs_plain_predictor(df : pd.DataFrame) -> pd.DataFrame:
    
    X = df[["soil_moisture_40", "steps_from_peak", "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c"]]

    return X

def separate_slopes_by_groups(df : pd.DataFrame) -> pd.DataFrame:
    df_decay = df.copy()
    restricted_mask = ~df_decay["plain"] & ~df["up"]
    group_start = restricted_mask & ~restricted_mask.shift(fill_value=False)
    df_decay["group"] = group_start.cumsum()
    df_decay.loc[~restricted_mask, "group"] = -1
    df_decay["group"] = df_decay["group"].astype(int)

    
    return df_decay

def compute_group_slopes(df : pd.DataFrame) -> pd.DataFrame:
    
    df_decay_grouped = df.copy() 

    num_groups = df_decay_grouped["group"].unique()[-1]

    df_decay_grouped["slope"] = 0

    for i in range(1, num_groups+1):

        y = df_decay_grouped.loc[df_decay_grouped["group"] == i, "soil_moisture_40"].values
        x = np.array([i for i in range(len(y))])

        if len(y) > 3:

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

            df_decay_grouped.loc[df_decay_grouped["group"] == i, "slope"] = m

        else:
            df_decay_grouped.loc[df_decay_grouped["group"] == i, "plain"] = True
            df_decay_grouped.loc[df_decay_grouped["group"] == i, "group"] = -1
        
    return df_decay_grouped

def compute_slopes_dataset(df : pd.DataFrame) -> pd.DataFrame:

    df_decay_grouped = df.copy()

    df_slopes = df_decay_grouped.loc[df_decay_grouped["group"] > 1]
    first_three_per_group = df_slopes.groupby('group').apply(lambda x: x.iloc[:3]).reset_index(drop=True).copy()
    first_three_per_group['point_idx'] = first_three_per_group.groupby('group').cumcount() + 1
    wide_values = first_three_per_group.pivot(index='group', columns='point_idx', values='soil_moisture_40')
    wide_values.columns = ['moisture_1', 'moisture_2', 'moisture_3']

    first_row_vars = df_slopes.groupby('group').first().reset_index()
    final_df = first_row_vars.merge(wide_values.reset_index(), on='group')

    final_df = final_df[["date", "moisture_1", "moisture_2", "moisture_3" , "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c", "slope"]]

    X = final_df[["moisture_1", "moisture_2", "moisture_3" , "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c"]]
    y = final_df[["slope"]]

    return X, y