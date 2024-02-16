import math
import pandas as pd
import numpy as np
from prophet import Prophet

def train(df):
    df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df_prophet = df.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
    
    train = df_prophet[:800]
    test = df_prophet[801:]
    
    model = Prophet(interval_width=0.99)
    model.fit(train)
    
    forecastlen = len(test)
    future = model.make_future_dataframe(periods=forecastlen,freq ='h')
    forecast = model.predict(future)
    
    test_data_forecasts = np.array(forecast["yhat"][800:])
    ground_truth = test.y
    
    errors = np.abs(ground_truth-test_data_forecasts)

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
    (mae,mse) = train(df)
    maes.append(mae)
    mses.append(mse)

maes_np = np.array(maes)
mses_np = np.array(mses)

mean_mae = maes_np.mean()
mean_mse = mses_np.mean()

print(f"MAE: {mean_mae}")
print(f"MSE: {mean_mse}")