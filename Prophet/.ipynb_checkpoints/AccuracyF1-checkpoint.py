import math
from prophet import Prophet
import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
    

def synth_data(df):
    df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    
    base_series = df.value
    time = np.arange(0, len(base_series), 1) 
    start_index = math.floor((1/2)*len(df.value))
    
    proportion_anomalies = 0.007
    num_anomalies = int((len(time) - start_index) * proportion_anomalies) # this will be mutltiplied with 2 since there will be upwards and downwards point anomalies
    
    anomalous_series = base_series.copy()
    
    #Positive Point Anomalies
    p_anomaly_indices = np.random.choice(range(start_index, len(time)), num_anomalies, replace=False)
    anomaly_amplitudes = np.random.uniform(0.7, 2.0, num_anomalies)
    
    anomalous_series.iloc[p_anomaly_indices] += anomaly_amplitudes
    
    #Negative Point Anomlaies
    n_anomaly_indices = np.random.choice(range(start_index, len(time)), num_anomalies, replace=False)
    anomaly_amplitudes = np.random.uniform(-2.0, -0.7, num_anomalies)
    
    anomalous_series.iloc[n_anomaly_indices] += anomaly_amplitudes
    
    anomaly_indices = np.append(p_anomaly_indices, n_anomaly_indices)
        
    df['Anomalie_Data'] = anomalous_series
    df['Anomalous'] = 0
    df.iloc[anomaly_indices, df.columns.get_loc('Anomalous')] = 1

    drift_index = math.floor((2/3)*len(anomalous_series))
    df.iloc[drift_index, df.columns.get_loc('Anomalous')] = 1
    df = df.drop(["value"], axis=1)
    df_prophet = df.reset_index().rename(columns={'index': 'ds', 'Anomalie_Data': 'y'})

    return df_prophet


def train_and_evaluate(df):
    train = df[:800]
    test = df[801:]
    
    model = Prophet(interval_width=0.99)
    model.fit(train)
    
    forecastlen = len(test)
    
    future = model.make_future_dataframe(periods=forecastlen,freq ='h')
    forecast = model.predict(future)
    
    ground_truth_values = test.y
    anomaly_scores = np.zeros(len(ground_truth_values))
    
    for i, true_value in enumerate(ground_truth_values):
        lower_bound = forecast["yhat_lower"][i]
        upper_bound = forecast["yhat_upper"][i] 
        prediction = forecast["yhat"][i]
        
        threshold = np.abs(prediction-upper_bound) * 10
        error = np.abs(true_value - prediction)
    
        if error >= threshold:
            anomaly_scores[i] = 1.0
        else:
            anomaly_scores[i] = error / threshold
    
    
    anomaly_scores = np.array(anomaly_scores)  
    true_labels = np.array(df.Anomalous[801:])  
    
    thresholds = np.arange(0.0, 1.01, 0.01) 
    max_f1 = 0
    optimal_threshold = 0
    
    for threshold in thresholds:
        predicted_labels = np.where(anomaly_scores >= threshold, 1, 0)
        f1 = f1_score(true_labels, predicted_labels)
    
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold
    
    predicted_labels = np.where(anomaly_scores >= optimal_threshold, 1, 0)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return (accuracy, tp, fp, tn, fn, recall, precision, f1)


df = pd.read_csv("../Data/RawTrafficData.csv", index_col=0)
df.index = pd.to_datetime(df.index)

repetitions = 100
accuracies = []
tps = []
fps = []
tns = []
fns = []
recalls = []
precisions = []
f1s = []

for i in range(repetitions):
    anomalous = synth_data(df)
    (accuracy, tp, fp, tn, fn, recall, precision, f1)= train_and_evaluate(anomalous)
    
    accuracies.append(accuracy)
    
    tps.append(tp)
    fps.append(fp)
    tns.append(tn)
    fns.append(fn)
    
    recalls.append(recall)
    precisions.append(precision)
    f1s.append(f1)

accuracies = np.array(accuracies)

tps = np.array(tps)
fps = np.array(fps)
tns = np.array(tns)
fns = np.array(fns)

recalls = np.array(recalls)
precisions = np.array(precisions)
f1s = np.array(f1s)


print(f"Accuracy: {accuracies.mean()}")
print(f"TP: {tps.mean()}, FP: {fps.mean()}, TN: {tns.mean()}, FN: {fns.mean()}")
print(f"Recall: {recalls.mean()}, Precision: {precisions.mean()}, F1 Score: {f1s.mean()}")

