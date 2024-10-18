import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from utils import MLP, LogitsToPredicate, DataLoader
import custom_fuzzy_ops as custom_fuzzy_ops
import logging
import sys

# Set up logging
log_file = "/home/zyang44/Github/baseline_cicIOT/LTNtorch/LTN_IoMT/training_log.txt"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)  # This allows printing to both console and log file
                    ])

#####################Utils#################################
def compute_sat_level(loader):
    mean_sat = 0
    for data, label_L1, label_L2 in loader:
        x = ltn.Variable("x", data)
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 0])
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 1])
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 2])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 3])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 4])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 5])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 6])
        x_benign = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 7])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_MQTT, l_MQTT),
            (x_Benign, l_Benign),
            (x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
            (x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
            (x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
            (x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
            (x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
            (x_benign, l_benign)
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
        mean_sat += SatAgg(*valid_forall_expressions)
    # In the loop: mean_sat accumulates the satisfaction levels for all the logical rules across the batches.
    # After the loop: mean_sat becomes the average satisfaction level of the logical rules over the entire dataset.
    mean_sat /= len(loader)
    return mean_sat

# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader, threshold=0.5):
    mean_accuracy = 0.0
    total_samples = 0
    for data, (label_L1, label_L2) in loader:
        # Get predictions from the MLP model
        predictions = mlp(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Assuming the model outputs two sets of predictions:
        # - Binary predictions for Label_L1 (1 value per sample)
        # - Multiclass predictions for Label_L2 (6 values per sample)
        pred_L1 = predictions[:, 0]  # Binary prediction for Label_L1
        pred_L2 = predictions[:, 1:]  # Multiclass predictions for Label_L2
        # For Label_L1 (binary classification)
        predicted_binary = (pred_L1 > threshold).astype(int)
        true_binary = label_L1.cpu().numpy()  # Convert tensor to numpy for comparison
        # For Label_L2 (multiclass classification)
        predicted_multiclass = np.argmax(pred_L2, axis=-1)
        true_multiclass = label_L2.cpu().numpy()  # Convert tensor to numpy
        
        # Compute binary accuracy for Label_L1
        binary_accuracy = np.mean(predicted_binary == true_binary)
        # Compute multiclass accuracy for Label_L2
        multiclass_accuracy = np.mean(predicted_multiclass == true_multiclass)
        # Combine the accuracy from both binary and multiclass predictions
        accuracy = (binary_accuracy + multiclass_accuracy) / 2
        
        # Accumulate mean accuracy over all batches
        mean_accuracy += accuracy
        total_samples += 1
    return mean_accuracy / total_samples

#####################Preprocess#################################
# 加载数据集
processed_train_file = '../CIC_IoMT/19classes/filtered_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/filtered_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数
label_L1_mapping = {"MQTT": 0, "Benign": 1}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 2, "MQTT-DDoS-Publish_Flood": 3, 
                    "MQTT-DoS-Connect_Flood": 4, "MQTT-DoS-Publish_Flood": 5,
                    "MQTT-Malformed_Data": 6, "Benign": 7}
train_label_L1, train_label_L2 = train_data.pop("label_L1"), train_data.pop("label_L2"),
test_label_L1, test_label_L2 = test_data.pop("label_L1"), test_data.pop("label_L2")

# 使用 StandardScaler 对数据进行缩放
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 将缩放后的数据和标签转换为Tensor
# 定义设备并移动数据和标签到设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torch.tensor(train_data_scaled).float().to(device)
test_data = torch.tensor(test_data_scaled).float().to(device)
train_label_L1 = torch.tensor(train_label_L1.to_numpy()).long().to(device)
train_label_L2 = torch.tensor(train_label_L2.to_numpy()).long().to(device)
test_label_L1 = torch.tensor(test_label_L1.to_numpy()).long().to(device)
test_label_L2 = torch.tensor(test_label_L2.to_numpy()).long().to(device)

print("Data processing and scaling done.")

#####################Setting#################################
# we define the constants
l_MQTT = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]))
l_Benign = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]))
l_MQTT_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]))
l_MQTT_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]))
l_MQTT_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]))
l_benign = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]))

# 创建模型实例并移动到设备 
mlp = MLP(layer_sizes=(45, 64, 32, 8)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
P = ltn.Predicate(LogitsToPredicate(mlp))

# define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(custom_fuzzy_ops.AndProd())
# And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

print("LTN setting done.")


#####################Training#################################
# create train and test loader (train_sex_labels, train_color_labels)
batch_size = 64
train_loader = DataLoader(train_data, (train_label_L1, train_label_L2), batch_size, shuffle=True)
test_loader = DataLoader(test_data, (test_label_L1, test_label_L2), batch_size, shuffle=False)