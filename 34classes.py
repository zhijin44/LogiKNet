import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_DIRECTORY = 'CIC_IOT/0.001percent_34classes.csv'

X_columns = [
    'flow_duration', 'header_length', 'protocol_type', 'duration',
    'rate', 'srate', 'drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'http', 'https', 'dns', 'telnet', 'smtp', 'ssh', 'irc', 'tcp',
    'udp', 'dhcp', 'arp', 'icmp', 'ipv', 'llc', 'tot_sum', 'min',
    'max', 'avg', 'std', 'tot_size', 'iat', 'number', 'magnitude',
    'radius', 'covariance', 'variance', 'weight',
]
y_column = 'label'

# Preparing Data
SEED = 42
training_data, test_data = train_test_split(pd.read_csv(DATASET_DIRECTORY), test_size=0.2, random_state=SEED)


# Scaling
scaler = StandardScaler()
scaler.fit(training_data[X_columns])


# Training
ML_models = [
    LogisticRegression(n_jobs=-1),
]

ML_names = [
    "LogisticRegression",
]


training_data[X_columns] = scaler.transform(training_data[X_columns])
for model in ML_models:
    model.fit(training_data[X_columns], training_data[y_column])
del training_data


# Testing
test_data[X_columns] = scaler.transform(test_data[X_columns])

predictions = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_matrices = []

for model, name in zip(ML_models, ML_names):
    # Predict the labels for the test data
    y_pred = model.predict(test_data[X_columns])
    predictions.append(y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_data[y_column], y_pred)
    precision = precision_score(test_data[y_column], y_pred, average='macro')
    recall = recall_score(test_data[y_column], y_pred, average='macro')
    f1 = f1_score(test_data[y_column], y_pred, average='macro')
    confusion = confusion_matrix(test_data[y_column], y_pred)

    # Store the metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)

    # Optionally, print the evaluation metrics for each model
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion}\n")
