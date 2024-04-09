import pandas as pd
import os
from tqdm import tqdm
from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 定义数据集目录
TRAIN_DIR = 'CIC_IoMT/train/'
TEST_DIR = 'CIC_IoMT/test/'

# processed_train_file = 'CIC_IoMT/6classes/processed_train_data_6classes.csv'
# processed_test_file = 'CIC_IoMT/6classes/processed_test_data_6classes.csv'
processed_train_file = 'CIC_IoMT/6classes/6classes_15k_train.csv'
processed_test_file = 'CIC_IoMT/6classes/6classes_1700_test.csv'

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


# 数据读取和标准化函数，匹配文件名中的关键词
def load_and_preprocess_data(directory):
    print(f"Loading data from {directory}...")
    data_frames = []
    for filename in tqdm(os.listdir(directory), desc="Files"):
        if filename.endswith('.csv'):
            label = None
            # Check for each specific label in the filename
            if "ARP_Spoofing" in filename:
                label = "ARP_Spoofing"
            elif "Benign" in filename:
                label = "Benign"
            elif "MQTT" in filename:
                label = "MQTT"
            elif "Recon" in filename:
                label = "Recon"
            # Adjusted conditions for TCP_IP-DDOS and TCP_IP-DOS recognition
            elif "TCP_IP-DDoS" in filename:
                label = "TCP_IP-DDOS"
            elif "TCP_IP-DoS" in filename:
                label = "TCP_IP-DOS"

            if label:
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                df['label'] = label
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
    training_data.to_csv(processed_train_file, index=False)
    test_data.to_csv(processed_test_file, index=False)
    print("Processed data saved.")

print("Scaling data...")
scaler = StandardScaler()
scaler.fit(training_data[X_columns])
training_data[X_columns] = scaler.transform(training_data[X_columns])
test_data[X_columns] = scaler.transform(test_data[X_columns])
# After scaling your training data in the training script
# scaler_save_path = 'CIC_IoMT/6classes/scaler_6classes.joblib'
scaler_save_path = 'CIC_IoMT/6classes/scaler_6classes_reduced.joblib'
dump(scaler, scaler_save_path)
print("Data scaled.")

# Check if the model already exists
# model_save_path = 'CIC_IoMT/6classes/mlp_classifier_model_6classes.joblib'
model_save_path = 'CIC_IoMT/6classes/mlp_classifier_model_6classes_reduced.joblib'
if os.path.exists(model_save_path):
    print("Loading existing model...")
    model = load(model_save_path)
else:
    print("Training model...")
    # Model training code
    model = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=200, activation='relu', solver='adam', random_state=1,
                          verbose=True)
    model.fit(training_data[X_columns], training_data['label'])
    # Save the trained model
    dump(model, model_save_path)
    print("Model saved to", model_save_path)

print("Making predictions...")
y_test = test_data['label']
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

