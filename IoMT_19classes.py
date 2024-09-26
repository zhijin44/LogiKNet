import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 定义数据集目录
TRAIN_DIR = 'CIC_IoMT/train/'
TEST_DIR = 'CIC_IoMT/test/'

# 定义处理后的数据文件路径
processed_train_file = 'CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = 'CIC_IoMT/19classes/reduced_test_data.csv'
# processed_train_file = 'CIC_IoMT/19classes/processed_train_data.csv'
# processed_test_file = 'CIC_IoMT/19classes/processed_test_data.csv'

# 特征列名称
X_columns = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

print("Starting data loading and preprocessing...")


# 标签提取函数
def extract_label(filename):
    label = filename.replace('.pcap.csv', '')
    label = re.sub(r'[_\d]+(train|test)$', '', label)
    return label


# 读取和标准化数据集
def load_and_preprocess_data(directory):
    print(f"Loading data from {directory}...")
    data_frames = []
    for filename in tqdm(os.listdir(directory), desc="Files"):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df['label'] = extract_label(filename)
            data_frames.append(df)
    full_data = pd.concat(data_frames, ignore_index=True)
    print("Data loaded and concatenated.")
    return full_data


# 数据预处理
if os.path.exists(processed_train_file) and os.path.exists(processed_test_file):
    print("Loading processed data...")
    training_data = pd.read_csv(processed_train_file)
    test_data = pd.read_csv(processed_test_file)
else:
    print("Processed data not found, loading and preprocessing raw data...")
    # 在这里调用 load_and_preprocess_data 函数以及其他任何必要的数据预处理步骤
    training_data = load_and_preprocess_data(TRAIN_DIR)
    test_data = load_and_preprocess_data(TEST_DIR)

    # 保存处理好的训练数据和测试数据
    processed_train_file = 'CIC_IoMT/19classes/reduced_train_data.csv'
    processed_test_file = 'CIC_IoMT/19classes/reduced_test_data.csv'
    # processed_train_file = 'CIC_IoMT/19classes/processed_train_data.csv'
    # processed_test_file = 'CIC_IoMT/19classes/processed_test_data.csv'
    training_data.to_csv(processed_train_file, index=False)
    test_data.to_csv(processed_test_file, index=False)
    print("Processed data saved.")


print("Scaling data...")
scaler = StandardScaler()
scaler.fit(training_data[X_columns])
training_data[X_columns] = scaler.transform(training_data[X_columns])
test_data[X_columns] = scaler.transform(test_data[X_columns])
scaler_save_path = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/scaler_reduced_baseline.joblib'
joblib.dump(scaler, scaler_save_path)
print("Data scaled.")


# Check if the model already exists
model_save_path = 'CIC_IoMT/19classes/mlp_classifier_reduced_model.joblib'
# model_save_path = 'CIC_IoMT/19classes/mlp_classifier_model.joblib'
if os.path.exists(model_save_path):
    print("Loading existing model...")
    model = joblib.load(model_save_path)
else:
    print("Training model...")
    # Model training code
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, activation='relu', solver='adam', random_state=1, verbose=True)
    model.fit(training_data[X_columns], training_data['label_L2'])
    # model.fit(training_data[X_columns], training_data['label'])
    # Save the trained model
    joblib.dump(model, model_save_path)
    print("Model saved to", model_save_path)

print("Making predictions...")
y_test = test_data['label_L2']
# y_test = test_data['label']
y_pred = model.predict(test_data[X_columns])
print("Predictions made.")


print("Evaluating model...")
# 计算和打印评估指标
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Recall Score (Macro):', recall_score(y_test, y_pred, average='macro'))
print('Precision Score (Macro):', precision_score(y_test, y_pred, average='macro'))
print('F1 Score (Macro):', f1_score(y_test, y_pred, average='macro'))
# print('Recall Scores by Class:', recall_score(y_test, y_pred, average=None))
# print('Precision Scores by Class:', precision_score(y_test, y_pred, average=None))
# print('F1 Scores by Class:', f1_score(y_test, y_pred, average=None))
print("Model evaluation complete.")
