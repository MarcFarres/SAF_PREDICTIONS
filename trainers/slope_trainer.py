import pandas as pd
import numpy as np
from utils import preprocess
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'  
    
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

    dtrain = xgb.DMatrix(X_train, label=y_train)
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


if __name__ == "__main__":

    THRESH_DOWN = -0.1
    THRESH_UP = 0.1

    df = pd.read_csv("data/1082-Device-Data-Fix.csv")
    df = preprocess.get_clean_df(df)
    df = preprocess.create_standarized_gradients(df)
    df = preprocess.create_steps_from_irrigation(df)

    plain_dates = preprocess.get_plains_dates(df, THRESH_DOWN, THRESH_UP)
    df["plain"] = False
    df.loc[df["date"].isin(plain_dates), "plain"] = True
    df_decay = preprocess.get_dataset_from_df(df, THRESH_UP)
    df_decay.head()

    X = df_decay[["soil_moisture_40", "steps_from_peak", "season_autumn", "season_spring", "season_summer", "season_winter", "hour_s", "hour_c"]]
    y = df_decay["plain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print("Best trial:")
    print("  Value (AUC):", study.best_value)
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    print("Best hyperparameters:", study.best_params)
    print("Best loss:", study.best_value)

    best_parameters = study.best_params.copy()

    test_model = xgb.XGBClassifier(**best_parameters)
    test_model.fit(X_train, y_train)

    y_proba = test_model.predict_proba(X_test)[:, 1]
    y_pred = test_model.predict(X_test)

    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    final_model = xgb.XGBClassifier(**best_parameters)
    final_model.fit(X, y)

    joblib.dump(final_model, "models/XGBoost_plain_classifier.joblib")