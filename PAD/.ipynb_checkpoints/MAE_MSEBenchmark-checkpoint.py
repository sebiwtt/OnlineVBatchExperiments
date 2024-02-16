import math
from river import anomaly
from river import time_series
from river import preprocessing
from river import linear_model
from river import optim
import pandas as pd
import numpy as np

def train(rawdata):

    period = 24
    horizon = 1
    
    predictive_model = time_series.SNARIMAX(
        p=6,
        d=0,
        q=6,
        m=period,
        sd=1,
        sq = 2,
        sp = 2,
        regressor=(
            preprocessing.StandardScaler()
            | linear_model.LinearRegression(
                optimizer=optim.SGD(0.001),
                l2 = 0.1,
                intercept_lr=.0000000001
            )
        ),
    )
    
    PAD = anomaly.PredictiveAnomalyDetection(
        predictive_model,
        horizon=1,
        n_std=3.0,
        warmup_period=period*1/3
    )
    
    scores = []
    predictions = []
    errors = []
    thresholds = []
    
    for y in rawdata:
    
        score, prediction, error, threshold = PAD.score_one_detailed(None, y)
    
        scores.append(score)
        thresholds.append(threshold)
        errors.append(error)
        predictions.append(prediction)
    
        PAD = PAD.learn_one(None, y)

    errors = np.abs(rawdata - predictions)
    errors = errors[800:] #Only counting from 800 for a fair comparison, ARIMA and Prophet get the first 800 instances for training and the rest as test set for evaluation

    squared_errors = np.square(errors)
    
    mae = errors.mean()
    mse = squared_errors.mean()

    return (mae,mse)

df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
df.index = pd.to_datetime(df.index)

repetitions = 100
maes = []
mses = []

for i in range(repetitions):
    (mae,mse) = train(df.value)
    maes.append(mae)
    mses.append(mse)

maes_np = np.array(maes)
mses_np = np.array(mses)

mean_mae = maes_np.mean()
mean_mse = mses_np.mean()

print(f"MAE: {mean_mae}")
print(f"MSE: {mean_mse}")