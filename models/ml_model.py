import pandas as pd
import xgboost as xgb
from utils import preprocess
import numpy as np
import joblib
import os
import logging
from typing import List, Union, Tuple
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

class MLModel():
    
    REQUIRED_COLUMNS = ["date", "soil_moisture_40"]
    SEASON_ENUM = {"autumn" : 0, "spring" : 1, "summer" : 2, "winter" : 3}
    
    def __init__(self):
        
        self.regressor = None
        
    def _check_dataframe(self, df : pd.DataFrame):
        
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            raise ValueError(f"Missing required columns: {self.REQUIRED_COLUMNS}")
    
    def _get_steps(self, df : pd.DataFrame) -> pd.DataFrame:
        
        df_steps = df
        
        orig_values = df_steps["soil_moisture_40"].to_numpy()
        values = gaussian_filter1d(orig_values, sigma=2)
        df_steps["soil_moisture_40"] = values
        
        df_steps["prev_value"] = df_steps["soil_moisture_40"].shift(1)
        df_steps['difference'] = df_steps['soil_moisture_40'] - df_steps['prev_value']
        df_steps["peak"] = False
        df_steps["steps_from_peak"] = 0
        
        current_steps = -1
        up = False
        
        for row in df_steps.iloc[1:].itertuples(index=True):
            down = row.difference <= 0
            current_steps += 1
            
            if not down:
                df_steps.loc[row.Index, "peak"] = True
                current_steps = -1
            else:
                df_steps.loc[row.Index, "steps_from_peak"] = current_steps
                
        df_steps["soil_moisture_40"] = orig_values
        
        return df_steps
    
    def _get_decays(self, df : pd.DataFrame) -> pd.DataFrame:
        df_decay = df[~df["peak"]]
        df_decay = df_decay.drop(columns=["irrigation_volume_0", "irrigation_volume_accumulated_0", "prev_value", "difference", "peak"])[1:].reset_index(drop=True)
        df_decay["hour"] = df_decay["date"].dt.hour
        df_decay["season"] = df_decay["date"].dt.month.apply(preprocess.get_season)
        df_decay = pd.get_dummies(df_decay, columns=['season'])
        df_decay['soil_moisture_next'] = df_decay['soil_moisture_40'].shift(-1)
        df_decay['next_steps'] = df_decay['steps_from_peak'].shift(-1)
        df_decay = df_decay[df_decay["next_steps"] != 0]
        df_decay = df_decay.drop(columns=["date", "soil_moisture_20", "soil_moisture_60", "next_steps"])[:-1]
        
        return df_decay
    
    def _format_training_data(self, X : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train = X
        
        X_train["Input_0"] = X_train["soil_moisture_40"]
        X_train["Output_1"] = X_train["soil_moisture_40"].shift(-1)
        X_train["Output_2"] = X_train["soil_moisture_40"].shift(-2)
        X_train["Output_3"] = X_train["soil_moisture_40"].shift(-3)
        X_train["season_autumn"] = X_train["season_autumn"].astype(bool)
        X_train["season_summer"] = X_train["season_summer"].astype(bool)
        X_train["season_spring"] = X_train["season_spring"].astype(bool)
        X_train["current_step"] = X_train["steps_from_peak"].shift(-3)
        X_train["hour"] = X_train["hour"]
        X_train["hour_s"] = np.sin(2 * np.pi * X_train["hour"])
        X_train["hour_c"] = np.cos(2 * np.pi * X_train["hour"])
        condition = X_train["current_step"] >= 3
        condition_full = ~X_train["Output_3"].isna() & condition
        X_train = X_train.loc[condition_full]

        y_train = X_train[["Output_1", "Output_2", "Output_3"]]
        X_train = X_train[["Input_0", "current_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]]
        
        return X_train, y_train
   
    def _get_model_input(self, current_moisture : float, current_step : int, current_date : pd.Timestamp) -> pd.DataFrame:
    
        seasons = [False]*3
        
        season = preprocess.get_season(current_date.month)
        if season != "winter":
            seasons[self.SEASON_ENUM[season]] = True
        
        hour_s = np.sin(2 * np.pi * current_date.hour)
        hour_c = np.cos(2 * np.pi * current_date.hour)
        
        model_input = [[current_moisture],
                        [current_step],
                        [hour_s, hour_c],
                        seasons,
                        ]
        
        model_input = np.concatenate(model_input)
        model_input = pd.DataFrame(model_input.reshape(1, -1), columns=["Input_0", "current_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"])
        
        return model_input
   
    def load_model(self, path : str):
        """
        Loads the model from a given file. It only accept XGBRegressors.
        
        Parameters
        ----------
        path : str
            The path to the file containing the model. It must be a joblib containing an `XGBRegressor`
        """
        
        regressor = joblib.load(path)
        
        if isinstance(regressor, xgb.XGBRegressor):
            self.regressor = regressor
            logger.info("Model loaded correctly")
            return
        
        raise TypeError(f"Expected XGBRegressor, got {type(regressor).__name__}")
   
    def train(self, df : pd.DataFrame, save_model : bool = False, model_name : str = "MLModel"):
        """
        Trains the model with the given dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that must contain at least soil_moisture_40 and its associated timestamps (`[soil_moisture_40, date]`)
        save_model : bool, optional
            Whether to save the trained model or not. It will be saved in `models/weights/MLModel.joblib`. (Default is `False`)
        model_name : str, optional
            The name you want to save the model with. (Default is 'MLModel')
        """
        self._check_dataframe(df)
        
        self.regressor = xgb.XGBRegressor()
        
        df_train = self._get_steps(df)
        df_train = self._get_decays(df_train)
        
        train_data = df_train[["soil_moisture_40", "steps_from_peak", "hour", "season_autumn", "season_spring", "season_summer"]]
        train_data["hour"] = train_data["hour"] / 24
        
        X_train, y_train = self._format_training_data(train_data)
        self.regressor.fit(X_train, y_train)
        train_pred = self.regressor.predict(X_train)
        train_error = mean_absolute_error(y_train, train_pred)
        
        if save_model:
            os.makedirs("models/weights", exist_ok=True)
            joblib.dump(self.regressor, f"models/weights/{model_name}.joblib")
        
        logger.info(f"Trained with training error : {train_error}")
        
    def predict_steps(self, previous_data : Union[List[float], np.ndarray], current_date : pd.Timestamp, current_step : int, future_steps : int) -> List[float]:
        """
        Forecasts the values of soil moisture for the given previous data.
        
        Parameters
        ----------
        previous_data : List[float] or np.ndarray
            A list or array containing all the previous points to be considered. They must be points outside irrigation phase.
        current_date : pd.Timestamp
            The date of the last point measured and given in `previous_data`
        current_step : int
            The number of steps away from the first detection. Usually equal to the length of the `previous_data` list.
        future_steps : int
            How many steps in the future need to be forecasted
            
        Returns
        --------
        predictions : List[float]
            A list containing all forecasted values

        """
        if self.regressor is None:
            raise ValueError("Model not trained nor loaded")
        
        if not isinstance(previous_data, (list, np.ndarray)):
            raise TypeError(f"Expected list or np.ndarray, got {type(previous_data).__name__}")
        
        if len(previous_data) < 3:
            raise ValueError(f"Previous data must have at least 3 elements, got {len(previous_data)}")
        
        predictions = []
        
        logger.info("Beggining prediction")
        model_input = self._get_model_input(previous_data[-1], current_step, current_date)
        
        for _ in range(1, future_steps+1, 3):
            
            predicted = self.regressor.predict(model_input)
            predictions.append(predicted.flatten())
            
            current_step += 3
            current_date += pd.Timedelta(minutes=90)
            
            model_input = self._get_model_input(predictions[-1][-1], current_step, current_date)
            
        predictions = np.concatenate(predictions)[:future_steps]
        
        logger.info("Prediction succeed")
            
        return predictions