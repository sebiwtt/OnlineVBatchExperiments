import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm


def train(df):
    train = df.value[:800] # All models are trained on the first 800 instances (online learning is only evaluated after instance nr 800 to have fair comparison)
    test = df.value[801:]
    
    p, d, q = 6, 0, 6
    P, D, Q, s = 2, 1, 2, 24 
    
    sarima_model = sm.tsa.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), initialization='approximate_diffuse')
    results = sarima_model.fit()
    forecast_steps = len(test)  
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.05)
    
    errors = np.abs(test - forecast_values)
    squared_errors = np.square(errors)
    
    mae = errors.mean()
    mse = squared_errors.mean()

    return (mae,mse)

df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
df.index = pd.to_datetime(df.index)

repetitions = 3
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