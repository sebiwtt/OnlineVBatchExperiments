import math
from multiprocessing import Process, Array
from time import sleep
import psutil
import math
import pandas as pd
import numpy as np
from prophet import Prophet

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

def train(df_prophet):
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
    
    ground_truth_values = test.y
    anomaly_scores = np.zeros(len(ground_truth_values))
    
    for i, true_value in enumerate(ground_truth_values):
        lower_bound = forecast["yhat_lower"][i]
        upper_bound = forecast["yhat_upper"][i] 
        prediction = forecast["yhat"][i]
        
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
    df_prophet = df.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
    
    p_ram = Process(target=ram_measure, args=(ram_arr,))
    p_cpu = Process(target=cpu_measure,  args=(cpu_arr,))
    p_simulation = Process(target=simulation, args=(df_prophet,))
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
    df.to_csv("./Prophet_cpu_usage.csv")

    df = pd.DataFrame({"value": ram})
    df = df.set_index('value')
    df.to_csv("./Prophet_ram_usage.csv")

