import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from utils import MLP, LogitsToPredicate, DataLoaderMulti
import custom_fuzzy_ops as custom_fuzzy_ops
import logging
import sys

# Set up logging
log_file = "/home/zyang44/Github/baseline_cicIOT/LTNtorch/LTN_IoMT/training_log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
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
        x_benign = ltn.Variable("x_benign", data[label_L2 == 7])

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
                # valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
                valid_forall_expressions.append(Forall(variable, P(variable, label)))

        # rules - L1 class exclusive for each other
        valid_forall_expressions.append(Forall(x, Not(And(P(x, l_MQTT), P(x, l_Benign)))))
        valid_forall_expressions.append(Forall(x, And(P(x, l_benign), P(x, l_Benign))))
        
        mean_sat += SatAgg(*valid_forall_expressions)
    # In the loop: mean_sat accumulates the satisfaction levels for all the logical rules across the batches.
    # After the loop: mean_sat becomes the average satisfaction level of the logical rules over the entire dataset.
    mean_sat /= len(loader)
    return mean_sat


def compute_metrics(loader, model):
    all_preds = []
    all_labels = []
    
    for data, _, label_L2 in loader:
        # Get predictions from the model
        predictions = model(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Predicted class for Label_L2 (multiclass classification)
        pred_L2 = np.argmax(predictions[:, 2:], axis=-1) + 2  # Shift range from [0, 5] to [2, 7]
        true_L2 = label_L2.cpu().numpy()  # Convert tensor to numpy

        # Accumulate predictions and true labels
        all_preds.extend(pred_L2)
        all_labels.extend(true_L2)
    
    # Compute metrics for each class
    precision = precision_score(all_labels, all_preds, labels=np.arange(2, 8), average=None)
    recall = recall_score(all_labels, all_preds, labels=np.arange(2, 8), average=None)
    f1 = f1_score(all_labels, all_preds, labels=np.arange(2, 8), average=None)

    return precision, recall, f1


# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    mean_accuracy_L1 = 0.0
    mean_accuracy_L2 = 0.0
    # total_samples = 0
    for data, label_L1, label_L2 in loader:
        # Get predictions from the MLP model
        predictions = mlp(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Predicted class for Label_L1 (binary classification)
        pred_L1 = np.argmax(predictions[:, 0:2], axis=-1)
        true_L1 = label_L1.cpu().numpy()  # Convert tensor to numpy for comparison

        # Predicted class for Label_L2 (multiclass classification)
        pred_L2 = np.argmax(predictions[:, 2:], axis=-1) + 2  # Shift range from [0, 5] to [2, 7]
        true_L2 = label_L2.cpu().numpy()  # Convert tensor to numpy

        # Compute binary accuracy for Label_L1
        accuracy_L1 = np.mean(pred_L1 == true_L1)
        # Compute multiclass accuracy for Label_L2
        accuracy_L2 = np.mean(pred_L2 == true_L2)

        # Accumulate mean accuracy over all batches
        mean_accuracy_L1 += accuracy_L1
        mean_accuracy_L2 += accuracy_L2
        # total_samples += 1
    # Return mean accuracies for Label_L1 and Label_L2
    mean_accuracy_L1 /= len(loader)
    mean_accuracy_L2 /= len(loader)

    return mean_accuracy_L1, mean_accuracy_L2

#####################Preprocess#################################
# 加载数据集
# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_tiny_train.csv'
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_data.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数
label_L1_mapping = {"MQTT": 0, "Benign": 1}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 2, "MQTT-DDoS-Publish_Flood": 3, 
                    "MQTT-DoS-Connect_Flood": 4, "MQTT-DoS-Publish_Flood": 5,
                    "MQTT-Malformed_Data": 6, "Benign": 7}
train_label_L1 = train_data.pop("label_L1").map(label_L1_mapping)
train_label_L2 = train_data.pop("label_L2").map(label_L2_mapping)
test_label_L1 = test_data.pop("label_L1").map(label_L1_mapping)
test_label_L2 = test_data.pop("label_L2").map(label_L2_mapping)

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
# And = ltn.Connective(custom_fuzzy_ops.AndProd())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

print("LTN setting done.")


#####################Training#################################
# create train and test loader (train_sex_labels, train_color_labels)
batch_size = 64
train_loader = DataLoaderMulti(train_data, (train_label_L1, train_label_L2), batch_size, shuffle=True)
test_loader = DataLoaderMulti(test_data, (test_label_L1, test_label_L2), batch_size, shuffle=False)

print("Create train and test loader done.")

#####################querying setting#####################################
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
def phi1(features):   # True, for all x, P(x, l_Benign) -> not P(x, l_MQTT), because the label_L1 exclusive
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), Not(P(x, l_MQTT))), p=5)

def phi2(features):  # False
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), (P(x, l_MQTT))), p=5)

def phi3(features):  # True
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_benign), (P(x, l_Benign))), p=5)

def phi4(features):  # False
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_benign), (P(x, l_MQTT))), p=5)


# it computes the satisfaction level of a formula phi using the given data loader (train or test)
def compute_sat_level_phi(loader, phi):
    mean_sat = 0
    for features, _, _ in loader:
        mean_sat += phi(features).value
    mean_sat /= len(loader)
    return mean_sat

print("Querying setting.")

print("Start training...")
optimizer = torch.optim.Adam(P.parameters(), lr=0.0001)

for epoch in range(41):
    train_loss = 0.0

    for batch_idx, (data, label_L1, label_L2) in enumerate(train_loader):
        optimizer.zero_grad()

        x = ltn.Variable("x", data)
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 0])
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 1])
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 2])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 3])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 4])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 5])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 6])
        x_benign = ltn.Variable("x_benign", data[label_L2 == 7])

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
                valid_forall_expressions.append(Forall(variable, P(variable, label)))

        # rules - L1 class exclusive for each other
        valid_forall_expressions.append(Forall(x, Not(And(P(x, l_MQTT), P(x, l_Benign)))))
        valid_forall_expressions.append(Forall(x, And(P(x, l_benign), P(x, l_Benign))))

        sat_agg = SatAgg(*valid_forall_expressions) # the satisfaction level over the current batch
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # print metrics
    if epoch % 1 == 0:
        train_sat = compute_sat_level(train_loader)
        test_sat = compute_sat_level(test_loader)
        train_acc = compute_accuracy(train_loader)
        test_acc = compute_accuracy(test_loader)
        print(f"epoch {epoch} | loss {train_loss:.4f} | Train Sat {train_sat:.3f} | Test Sat {test_sat:.3f} | "
          f"Train Acc L1 {train_acc[0]:.3f} | Train Acc L2 {train_acc[1]:.3f} | "
          f"Test Acc L1 {test_acc[0]:.3f} | Test Acc L2 {test_acc[1]:.3f}")
    
        logging.info(f"epoch {epoch} | loss {train_loss:.4f} | Train Sat {train_sat:.3f} | "
                 f"Test Sat {test_sat:.3f} | Train Acc L1 {train_acc[0]:.3f} | Train Acc L2 {train_acc[1]:.3f} | "
                 f"Test Acc L1 {test_acc[0]:.3f} | Test Acc L2 {test_acc[1]:.3f}")
    if epoch % 5 == 0:
        print(f"Test Sat Phi 1 {compute_sat_level_phi(test_loader, phi1):.3f} | Test Sat Phi 2 {compute_sat_level_phi(test_loader, phi2):.3f} | "
              f"Test Sat Phi 3 {compute_sat_level_phi(test_loader, phi3):.3f} | Test Sat Phi 4 {compute_sat_level_phi(test_loader, phi4):.3f}")
        logging.info(f"Test Sat Phi 1 {compute_sat_level_phi(test_loader, phi1):.3f} | Test Sat Phi 2 {compute_sat_level_phi(test_loader, phi2):.3f} | "
                     f"Test Sat Phi 3 {compute_sat_level_phi(test_loader, phi3):.3f} | Test Sat Phi 4 {compute_sat_level_phi(test_loader, phi4):.3f}")

#####################Evaluation#################################
class_names = [
    "MQTT-DDoS-Connect_Flood", 
    "MQTT-DDoS-Publish_Flood", 
    "MQTT-DoS-Connect_Flood", 
    "MQTT-DoS-Publish_Flood", 
    "MQTT-Malformed_Data",
    "benign"
]
precision, recall, f1 = compute_metrics(test_loader, mlp)
print(f"Macro Recall: {recall.mean():.4f}, Macro Precision: {precision.mean():.4f}, Macro F1-Score: {f1.mean():.4f}")
print("Scores by Class:")

logging.info(f"Macro Recall: {recall.mean():.4f}, Macro Precision: {precision.mean():.4f}, Macro F1-Score: {f1.mean():.4f}")
logging.info("Scores by Class:")

for i, class_name in enumerate(class_names):
    print(f"Class {class_name}: Recall: {recall[i]:.6f}, Precision: {precision[i]:.6f}, F1: {f1[i]:.6f}")
    logging.info(f"Class {class_name}: Recall: {recall[i]:.6f}, Precision: {precision[i]:.6f}, F1: {f1[i]:.6f}")
