import math
from multiprocessing import Process, Array
from time import sleep
import psutil
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm


def cpu_measure(a):
    cnt = 0
    while True:
        a[cnt] = psutil.cpu_percent()
        cnt += 1
        sleep(0.1)


def ram_measure(a):
    cnt = 0
    while True:
        a[cnt] = psutil.virtual_memory().percent
        cnt += 1
        sleep(0.1)

def train(df):
    train = df.value[:800]
    test = df.value[801:]
    
    p, d, q = 6, 0, 6
    P, D, Q, s = 2, 1, 2, 24 
    
    sarima_model = sm.tsa.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), initialization='approximate_diffuse')
    results = sarima_model.fit()
    forecast_steps = len(test)  
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.05)
    
    ground_truth_values = test
    anomaly_scores = np.zeros(len(ground_truth_values))
    
    for i, true_value in enumerate(ground_truth_values):
        lower_bound = confidence_intervals.iloc[i, 0] 
        upper_bound = confidence_intervals.iloc[i, 1]
        prediction = (lower_bound + upper_bound) / 2
        
        threshold = np.abs(prediction-upper_bound) * 6
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold


def simulation(data):
    repetitions = 100
    for i in range(repetitions):
        train(data)
        
if __name__ == '__main__':
    cpu_arr = Array('f', 1000)
    ram_arr = Array('f', 1000)
    df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    p_ram = Process(target=ram_measure, args=(ram_arr,))
    p_cpu = Process(target=cpu_measure,  args=(cpu_arr,))
    p_simulation = Process(target=simulation, args=(df,))
    p_ram.start()
    p_cpu.start()

    p_simulation.start()

    sleep(5)

    # p_simulation.join()
    p_ram.terminate()
    p_ram.join()
    p_cpu.terminate()
    p_cpu.join()

    cpu = np.array(cpu_arr[:])
    ram = np.array(ram_arr[:])
    ram = ram[ram != 0]
    cpu = cpu[cpu != 0]

    df = pd.DataFrame({"value": cpu})
    df = df.set_index('value')
    df.to_csv("./ARIMA_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./ARIMA_ram_usage.csv")

