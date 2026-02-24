import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def test_predict(model, format_func, steps=600, diff=True):
    current_value = 0.34
    current_hour = 10

    predictions = [(current_value, 0, current_hour)]

    input_names = ["soil_moisture_40", "steps_from_peak", "hour"]

    for i in range(steps):
        input_vec = np.array([predictions[-1][0], i, predictions[-1][2]/24]).reshape(1, -1)
        pred = model.predict(format_func(input_vec))
        new_hour = (predictions[-1][2] + i%2)%24
        if diff:
            predictions.append((pred[0] + predictions[-1][0], i+1, new_hour))
        else:
            predictions.append((predictions[-1][0], i+1, new_hour))
        
    print(predictions)

    values = [c[0] for c in predictions]
    times = [i for i in range(steps + 1)]

    sns.scatterplot(x=times, y=values)
    plt.show()

def test_real_predict(model, df, format_func, steps=600, diff=True):
    start_date = pd.to_datetime("2025-08-01")
    end_date   = pd.to_datetime("2025-08-30")
    df_pred = df[(df["date"]>= start_date) & (df["date"] <= end_date)].copy()

    current_value = df.loc[df["date"] == pd.to_datetime("2025-08-01 07:00:00"), "soil_moisture_40"].values[0]
    current_date = pd.to_datetime("2025-08-01 08:00:00")
    current_hour = 7

    predictions = [(current_value, 0, current_hour)]

    input_names = ["soil_moisture_40", "steps_from_peak", "hour"]

    for i in range(steps):
        input_vec = np.array([predictions[-1][0], i, predictions[-1][2]/24]).reshape(1, -1)
        pred = model.predict(format_func(input_vec))
        new_hour = (predictions[-1][2] + i%2)%24
        if diff:
            predictions.append((pred[0] + predictions[-1][0], i+1, new_hour))
        else:
            predictions.append((predictions[-1][0], i+1, new_hour))
        
    print(predictions)

    values = [c[0] for c in predictions]
    times = [current_date + pd.Timedelta(hours=i/2) for i in range(steps + 1)]

    plt.figure(figsize=(20, 10))
    sns.lineplot(df_pred, x="date", y="soil_moisture_40", label="depth40", color="red")
    sns.lineplot(x=times, y=values)
    plt.legend(title="Depth Variables")
    plt.ylim(0, 0.40)
    plt.yticks([0, 0.20, 0.40])
    plt.show()
    