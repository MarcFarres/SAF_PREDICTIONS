from logger import logging_formater
import argparse
import logging
from models import LinearModel, MLModel, CapacitanceDetector
from utils import preprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Usage example. In production you would want to accumulate seen points
and get the current step and date just by counting number of points 
and knowing the first date. With that you avoid redundant communications 
and save time and resources. When a new irrigation is done, the points are
reseted and you get again a new initial date to account for the irrigation
time. Could just stop communicating new points when irrigating and make
model reset after some time to avoid overheads, but that would make the
system non-reliable against possible communication delays, so I would
not recommend that.
"""

def setup_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(logging_formater.ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción de humedad del suelo")
    parser.add_argument(
        "--model", "-m",
        choices=["Linear", "ML"],
        default="ML",
        help="Modelo a usar: Linear o ML (default: ML)",
    )
    args = parser.parse_args()

    setup_logger()
    logger = logging.getLogger()

    df = pd.read_csv("data/1082-Device-Data-Fix.csv")
    df = preprocess.get_clean_df(df) #Here use whatever function needed to correctly format the data

    model = None

    if args.model == "Linear":
        #This training takes a bit of time since it searches for the best possible
        #parameters, so I'd recommend to train once and load everytime after that to
        #avoid any overhead.
        model = LinearModel()
        #model.train_plain_model(df.copy(), save_model=True)
        model.load_plain_model("models/weights/XGBoost_plain_classifier_new.joblib")
        logger.info("Selected linear model")
        
    elif args.model == "ML":
        #This gets trained really fast, so it is not really needed to store a model
        #however, it can be saved and loaded if there is the need.
        model = MLModel()
        model.train(df.copy(), save_model=False)
        #model.load_model("models/weights/MLModel.joblib")
        logger.info("Selected ML model")
        
    else:
        logger.ERROR("Non-existing model selected")
        exit()
        
    
    #Example date choosen for great visualization
    start_date = pd.to_datetime("2024-06-06 12:00:00")
    end_date   = pd.to_datetime("2024-06-11 05:00:00")
    
    #Get points after irrigation and last date before prediction
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    mask = df["irrigation_volume_0"] == 0
    first_point = mask[mask].index[0]
    df = df.loc[first_point:].copy()
    
    previous_values = df["soil_moisture_40"].iloc[:10].values
    """
    How these previous values are accumulated is really important for good
    performance of the models. If the ML model is used, then there is no need
    for any particular care. However, if the linear model is used, then it is
    important that points corresponding to plateaus are removed from the values
    given, since then it will mess up with the linear regression. Next there is
    a commented pseudocode that is an example of how to eliminate these kind of points
    when working with an accumulation of points.
    """
    
    # new_point_to_add = recieve_new_point()
    # grad = new_point_to_add - previous_values[-1]
    # if (grad < THRESHOLD):
    #     previous_values.append(new_point_to_add)

    """
    Another way to do this and try to not be too afected by sensor errors would
    be to remove points once you have that a given amount of points were detected
    as plateaus. You would allow some points on the data, but you would made sure
    the detection has actually been a plateau.
    """
    
    # new_point_to_add = recieve_new_point()
    # grad = new_point_to_add - previous_values[-1]
    # if (grad >= THRESHOLD):
    #     plain_detected += 1
    # else:
    #     plain_detected = 0
        
    # previous_values.append(new_point_to_add)
    
    # if (plain_detected > NUM_OF_DETECTIONS):
    #     previous_values = previous_values[:-1]
    
    """
    A final recomendation would be to use standarized gradients instead, so
    the values of the threshold are way more robust to changing the dataset.
    This way you can make a general rule to get rid of points. To do so a big
    amount of points should be stored to compute the standarized gradient with.
    Since even the full story of points of a sensor does not take too much
    memory, you could just have it all, and introduce the new points in it as
    they arrive. Otherwise, you could have just a year worth of detections to have
    the same results. Further reductions have not been tested by me, but it should
    work nicely even with just a month of data as far as I'm concerned.
    Of course, as one point arrives, the first point of the dataset can be erased, that
    way you always have the same amount of points.
    
    This is the approach I used when developing the models. The previous two
    are hard to tune for them to properly work, while this one allows for easier
    tunning, so I would personally use this one.
    
    The thresholds used in the development have been:
        THRESH_DOWN = -0.1
        THRESH_UP = 0.1
    """
    
    # new_point_to_add = recieve_new_point()
    # df_month = df_mont.add_new_point(new_point_to_add)
    # df_month = preprocess.create_standarized_gradients(df_month)
    # if not (df_month.loc[idx, "gradient_std"] > THRESH_DOWN) & (df_month.loc[idx, "gradient_std"] <= THRESH_DOWN):
    #     previous_values.append(df_month.iloc[-1, "soil_moisture_40"])
    
    
    
    current_date = df["date"].iloc[9]
    
    predictions = model.predict_steps(previous_values, current_date, 10, 100)
    
    correct_values = df["soil_moisture_40"].iloc[10:110].values
    
    steps = [i for i in range(110)]
    
    plt.scatter(steps[:10], previous_values, color="blue", label="Seen points")
    plt.plot(steps[10:], predictions, color="red", label="Predicted")
    plt.plot(steps[10:], correct_values, color="green", label="Real future values")
    plt.legend(loc="upper right")
    plt.ylabel("Soil moisture")
    plt.xlabel("Steps")
    plt.ylim(0.20, 0.36)
    plt.show()
    
    #Here we find capacitances. An anomaly detector or something like
    #that should be added into this in production, so big outliers do 
    #not hurt the user visualizaton when using it for normalization.
    capacitance_detector = CapacitanceDetector()
    capacitances = capacitance_detector.detect_capacitances(df)
    plt.plot(df["date"], df["soil_moisture_40"], label="Soil moisture")
    
    plt.axvline(capacitances["date"].iloc[0], color="red", label="Capacitance")
    for row in capacitances.iloc[1:].itertuples():
        plt.axvline(row.date, color="red")

    plt.legend(loc="upper right")
    plt.ylabel("Soil moisture")
    plt.xlabel("Date")
    plt.ylim(0.20, 0.36)
    plt.show()

    
        
    