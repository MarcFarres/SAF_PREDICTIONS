import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
import joblib
import xgboost as xgb
from utils import preprocess
import numpy as np
import optuna
import os
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class LinearModel():
    """
    A linear model to predict soil moisture decays mixed with an `XGBClassifier`
    for plateau detection.
    """
    
    THRESH_DOWN = -0.1
    THRESH_UP = 0.1
    REQUIRED_COLUMNS = ["date", "soil_moisture_40"]
    SEASON_ENUM = {"autumn" : 0, "spring" : 1, "summer" : 2, "winter" : 3}
    
    def __init__(self):
        
        self.regressor = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=False),
            min_samples=3,
            residual_threshold=1.5,
            random_state=0
        )
        
        self.plain_detector = None 
        
    def _check_dataframe(self, df : pd.DataFrame):
        
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            raise ValueError(f"Missing required columns: {self.REQUIRED_COLUMNS}")
        
    def _process_dataframe(self, df : pd.DataFrame) -> pd.DataFrame:
        df_processed = preprocess.create_standarized_gradients(df)
        df_processed = preprocess.create_steps_from_irrigation(df_processed)
        
        plain_dates = preprocess.get_plains_dates(df_processed, self.THRESH_DOWN, self.THRESH_UP)
        
        df_processed["plain"] = False
        df_processed.loc[df_processed["date"].isin(plain_dates), "plain"] = True
        
        return df_processed   
        
    def _optimize_plain_model(self, X : pd.DataFrame, y : pd.DataFrame) -> xgb.XGBClassifier:
        
        def objective(trial):

            params = {
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
            }

            dtrain = xgb.DMatrix(X, label=y)
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                nfold=3,
                num_boost_round=1000,
                early_stopping_rounds=20,
                stratified=True,
                seed=42,
                verbose_eval=False
            )

            mean_auc = cv_results['test-auc-mean'].max()

            return mean_auc
        
        logger.info("Beginning study for best parameters")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        logger.info("Best trial:")
        logger.info("  Value (AUC):", study.best_value)
        logger.info("  Params:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")

        logger.info("Best hyperparameters:", study.best_params)
        logger.info("Best loss:", study.best_value)

        best_parameters = study.best_params.copy()
        
        final_model = xgb.XGBClassifier(**best_parameters)
        final_model.fit(X, y)
        return final_model
        
    def train_plain_model(self, df : pd.DataFrame, save_model : bool = False, model_name : str = "XGBoost_plain_classifier"):
        """
        Trains the plain detector model with the given dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that must contain at least soil_moisture_40 and its associated timestamps (`[soil_moisture_40, date]`)
        save_model : bool, optional
            Whether to save the trained model or not. It will be saved in `models/weights/XGBoost_plain_classifier.joblib`. (Default is `False`)
        model_name : str, optional
            The name you want to save the model with. (Default is 'XGBoost_plain_classifier')
        """
        self._check_dataframe(df)
        df_processed = self._process_dataframe(df.copy())
        
        df_decay = preprocess.get_dataset_from_df(df_processed, self.THRESH_UP)
        
        X = df_decay[["soil_moisture_40", "steps_from_peak", "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c"]]
        y = df_decay["plain"]
        self.plain_detector = self._optimize_plain_model(X, y)
        
        if save_model:
            os.makedirs("models/weights", exist_ok=True)
            joblib.dump(self.plain_detector, f"models/weights/{model_name}.joblib")
            
        logger.info("Model trained")
    
    def load_plain_model(self, path : str):
        """
        Loads the plain detector model from a given file. It only accept XGBClassifiers.
        
        Parameters
        ----------
        path : str
            The path to the file containing the model. It must be a joblib containing an `XGBClassifier`
        """
        plain_detector = joblib.load(path)
        
        if isinstance(plain_detector, xgb.XGBClassifier):
            self.plain_detector = plain_detector
            logger.info("Plain model loaded correctly")
            return
        
        raise TypeError(f"Expected XGBClassifier, got {type(plain_detector).__name__}")
   
    def _line_regression(self, previous_data : List[float]) -> float:
        points = np.array(previous_data)
        x = np.array([i for i in range(len(points))])
        y = points
        x0, y0 = x[0], y[0]
        X = (x - x0).reshape(-1, 1)
        Y = y - y0
        self.regressor.fit(X, Y)
        m = self.regressor.estimator_.coef_[0]
        return m   
    
    def _get_plain_input(self, current_moisture : float, current_step : int, current_date : pd.Timestamp) -> pd.DataFrame:
        
        seasons = [False]*4
        seasons[self.SEASON_ENUM[preprocess.get_season(current_date.month)]] = True
        
        hour_s = np.sin(2 * np.pi * current_date.hour)
        hour_c = np.cos(2 * np.pi * current_date.hour)
        
        plain_input = [[current_moisture],
                       [current_step],
                        seasons,
                       [hour_s, hour_c]
                      ]
        plain_input = np.concatenate(plain_input)
        plain_input = pd.DataFrame(plain_input.reshape(1, -1), columns=["soil_moisture_40", "steps_from_peak", "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c"])
        
        return plain_input
        
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
        if self.plain_detector is None:
            raise ValueError("Plain detector not trained or loaded")
        
        if not isinstance(previous_data, (list, np.ndarray)):
            raise TypeError(f"Expected list or np.ndarray, got {type(previous_data).__name__}")
        
        if len(previous_data) < 3:
            raise ValueError(f"Previous data must have at least 3 elements, got {len(previous_data)}")
        
        m = self._line_regression(previous_data)
        b = previous_data[-1]
        
        predictions = []
        plain_steps = 0
        
        logger.info("Beggining prediction")
        
        plain_input = self._get_plain_input(previous_data[-1], current_step, current_date)
        
        for step in range(1, future_steps+1):
            
            if self.plain_detector.predict(plain_input):
                plain_steps += 1
                prev_pred = predictions[-1] if predictions else previous_data[-1]
                predictions.append(prev_pred)
            
            else:
                predictions.append(m*(step - plain_steps) + b)
                
            current_step += 1
            current_date += pd.Timedelta(minutes=30)
            plain_input = self._get_plain_input(predictions[-1], current_step, current_date)
            
        logger.info("Prediction succeed")
            
        return predictions[:future_steps]