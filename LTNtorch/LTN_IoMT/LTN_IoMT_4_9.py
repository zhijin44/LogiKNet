import ltn.fuzzy_ops
import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import MLP, LogitsToPredicate, DataLoaderMulti
import custom_fuzzy_ops as custom_fuzzy_ops
import logging
import sys
import os

# 特征列名称
X_columns = [
    'Header_Length', # 3 
    'Protocol Type', 'Duration', 
    'Rate', 
    'Srate', 'Drate',
    # 'fin_flag_number', 
    # 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    # 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'syn_count',
    'ack_count', # 5
    'fin_count', # 2
    # 'rst_count', 
    # 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    # 'SMTP', 'SSH', 'IRC', 
    # 'TCP', 
    # 'UDP', 'DHCP', 
    # 'ARP', 
    # 'ICMP', 'IGMP', 'IPv',
    # 'LLC', 'Tot sum', 'Min', 'Max', 
    # 'AVG', 
    # 'Std', 'Tot size', 
    'IAT', # 1
    # 'Number', 'Magnitue', 'Radius', 'Covariance', 
    # 'Variance', 
    # 'Weight' # 4
]


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
        x_Recon = ltn.Variable("x_Recon", data[label_L1 == 2])
        x_ARP_Spoofing = ltn.Variable("x_ARP_Spoofing", data[label_L1 == 3])
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 4])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 5])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 6])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 7])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 8])
        x_benign = ltn.Variable("x_benign", data[label_L2 == 9])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 10])
        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 11])
        x_arp_spoofing = ltn.Variable("x_arp_spoofing", data[label_L2 == 12])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_MQTT, l_MQTT),
            (x_Benign, l_Benign),
            (x_Recon, l_Recon),
            (x_ARP_Spoofing, l_ARP_Spoofing),
            (x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
            (x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
            (x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
            (x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
            (x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
            (x_benign, l_benign),
            (x_Recon_OS_Scan, l_Recon_OS_Scan),
            (x_Recon_Port_Scan, l_Recon_Port_Scan),
            (x_arp_spoofing, l_arp_spoofing)
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label)))

        # rules - L1 exclusive
        valid_forall_expressions.append(Exists(x, Not(And(P(x, l_MQTT), P(x, l_Benign), P(x, l_Recon), P(x, l_ARP_Spoofing)))))
        # rules - L2 exclusive
        valid_forall_expressions.append(Exists(x, Not(And(P(x, l_MQTT_DDoS_Connect_Flood), P(x, l_MQTT_DDoS_Publish_Flood), 
                                                          P(x, l_MQTT_DoS_Connect_Flood), P(x, l_MQTT_DoS_Publish_Flood),
                                                          P(x, l_MQTT_Malformed_Data), P(x, l_benign),
                                                          P(x, l_Recon_OS_Scan), P(x, l_Recon_Port_Scan),
                                                          P(x, l_arp_spoofing)))))
        # rules - hierarchy
        valid_forall_expressions.append(Exists(x, And(P(x, l_Benign), P(x, l_benign))))
        valid_forall_expressions.append(Exists(x, And(P(x, l_ARP_Spoofing), P(x, l_arp_spoofing))))

        mean_sat += SatAgg(*valid_forall_expressions)
    # In the loop: mean_sat accumulates the satisfaction levels for all the logical rules across the batches.
    # After the loop: mean_sat becomes the average satisfaction level of the logical rules over the entire dataset.
    mean_sat /= len(loader)
    return mean_sat

def compute_metrics(loader, model, class_names_L1, class_names_L2):
    all_preds_L1 = []
    all_labels_L1 = []
    all_preds_L2 = []
    all_labels_L2 = []
    
    for data, label_L1, label_L2 in loader:
        # Get predictions from the model
        predictions = model(data).detach().cpu().numpy()
        
        # Predicted and true classes for Label_L1
        pred_L1 = np.argmax(predictions[:, 0:4], axis=-1)
        true_L1 = label_L1.cpu().numpy()
        
        # Predicted and true classes for Label_L2
        pred_L2 = np.argmax(predictions[:, 4:], axis=-1) + 4  # Shift range to match [4, 12]
        true_L2 = label_L2.cpu().numpy()

        # Accumulate predictions and true labels for both labels
        all_preds_L1.extend(pred_L1)
        all_labels_L1.extend(true_L1)
        all_preds_L2.extend(pred_L2)
        all_labels_L2.extend(true_L2)
    
    # Classification report for Label_L1
    report_L1 = classification_report(
        all_labels_L1,
        all_preds_L1,
        target_names=class_names_L1,
        zero_division=0
    )
    print("Classification Report for Label_L1:\n")
    print(report_L1)
    
    # Classification report for Label_L2
    report_L2 = classification_report(
        all_labels_L2,
        all_preds_L2,
        target_names=class_names_L2,
        zero_division=0
    )
    print("\nClassification Report for Label_L2:\n")
    print(report_L2)
    
    return report_L1, report_L2

def compute_accuracy(loader):
    mean_accuracy_L1 = 0.0
    mean_accuracy_L2 = 0.0
    # total_samples = 0
    for data, label_L1, label_L2 in loader:
        # Get predictions from the MLP model
        predictions = mlp(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Predicted class for Label_L1 
        pred_L1 = np.argmax(predictions[:, 0:4], axis=-1)
        true_L1 = label_L1.cpu().numpy()  # Convert tensor to numpy for comparison

        # Predicted class for Label_L2 (multiclass classification)
        pred_L2 = np.argmax(predictions[:, 4:], axis=-1) + 4  # Shift range from [0, 8] to [4, 12]
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

def plot_confusion_matrix(loader, model, class_names, filename="4-9_LTN.png"):
    all_preds = []
    all_labels = []
    
    for data, _, label_L2 in loader:
        # Get predictions from the model
        predictions = model(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Predicted class for Label_L2 (multiclass classification)
        pred_L2 = np.argmax(predictions[:, 4:], axis=-1) + 4  # Shift range from [0, 8] to [4, 12]
        true_L2 = label_L2.cpu().numpy()  # Convert tensor to numpy

        # Accumulate predictions and true labels
        all_preds.extend(pred_L2)
        all_labels.extend(true_L2)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(4, 13))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='viridis', xticks_rotation=45)
    plt.title("Confusion Matrix")
    
    # Save the figure
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    print(f"Confusion matrix saved to {filepath}")

def plot_combined_confusion_matrices(loader, model, class_names_L1, class_names_L2, filename="4-9_LTN.png"):
    all_preds_L1 = []
    all_labels_L1 = []
    all_preds_L2 = []
    all_labels_L2 = []
    
    for data, label_L1, label_L2 in loader:
        # Get predictions from the model
        predictions = model(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        
        # Predicted class for Label_L1
        pred_L1 = np.argmax(predictions[:, 0:4], axis=-1)
        true_L1 = label_L1.cpu().numpy()  # Convert tensor to numpy

        # Predicted class for Label_L2 (multiclass classification)
        pred_L2 = np.argmax(predictions[:, 4:], axis=-1) + 4  # Shift range from [0, 8] to [4, 12]
        true_L2 = label_L2.cpu().numpy()  # Convert tensor to numpy

        # Accumulate predictions and true labels for both labels
        all_preds_L1.extend(pred_L1)
        all_labels_L1.extend(true_L1)
        all_preds_L2.extend(pred_L2)
        all_labels_L2.extend(true_L2)
    
    # Compute confusion matrices
    cm_L1 = confusion_matrix(all_labels_L1, all_preds_L1, labels=np.arange(0, 4))
    cm_L2 = confusion_matrix(all_labels_L2, all_preds_L2, labels=np.arange(4, 13))
    
    # Plot combined confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))  # Two subplots side by side

    # Plot confusion matrix for Label_L1
    disp_L1 = ConfusionMatrixDisplay(confusion_matrix=cm_L1, display_labels=class_names_L1)
    disp_L1.plot(ax=axs[0], cmap='viridis', xticks_rotation=45)
    axs[0].set_title("Confusion Matrix for Label_L1")

    # Plot confusion matrix for Label_L2
    disp_L2 = ConfusionMatrixDisplay(confusion_matrix=cm_L2, display_labels=class_names_L2)
    disp_L2.plot(ax=axs[1], cmap='viridis', xticks_rotation=45)
    axs[1].set_title("Confusion Matrix for Label_L2")

    # Save the combined figure
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    print(f"Combined confusion matrix saved to {filepath}")


#####################Preprocess#################################
# 加载数据集
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_m_4_9.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_4_9.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数
label_L1_mapping = {"MQTT": 0, "Benign": 1, "Recon": 2, "ARP_Spoofing": 3}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 4, "MQTT-DDoS-Publish_Flood": 5, 
                    "MQTT-DoS-Connect_Flood": 6, "MQTT-DoS-Publish_Flood": 7,
                    "MQTT-Malformed_Data": 8, "benign": 9, 
                    "Recon-OS_Scan": 10, "Recon-Port_Scan": 11,
                    "arp_spoofing": 12}
train_label_L1 = train_data.pop("label_L1").map(label_L1_mapping)
train_label_L2 = train_data.pop("label_L2").map(label_L2_mapping)
test_label_L1 = test_data.pop("label_L1").map(label_L1_mapping)
test_label_L2 = test_data.pop("label_L2").map(label_L2_mapping)

# 使用 StandardScaler 对数据进行缩放
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[X_columns])
test_data_scaled = scaler.transform(test_data[X_columns])

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
# Define the new constants with 13 classes (updated one-hot encoding for each label)
l_MQTT = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Benign = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_ARP_Spoofing = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
l_MQTT_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
l_benign = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
l_Recon_OS_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
l_Recon_Port_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
l_arp_spoofing = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

# define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(custom_fuzzy_ops.AndProd())
# And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
SatAgg = ltn.fuzzy_ops.SatAgg()

print("LTN setting done.")

#####################querying setting#####################################
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
def phi1(features):   # True 
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), (P(x, l_ARP_Spoofing))), p=5)

def phi2(features):  # True
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), (P(x, l_benign))), p=5)

def phi3(features):  # False
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), (P(x, l_Recon))), p=5)

def phi4(features):  # False
    x = ltn.Variable("x", features)
    return Forall(x, Implies(P(x, l_Benign), (P(x, l_ARP_Spoofing))), p=5)

# it computes the satisfaction level of a formula phi using the given data loader (train or test)
def compute_sat_level_phi(loader, phi):
    mean_sat = 0
    for features, _, _ in loader:
        mean_sat += phi(features).value
    mean_sat /= len(loader)
    return mean_sat

print("Querying setting.")

#####################Training#################################
# 创建模型实例并移动到设备 
mlp = MLP(layer_sizes=(9, 64, 32, 13)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
P = ltn.Predicate(LogitsToPredicate(mlp))

# create train and test loader (train_sex_labels, train_color_labels)
batch_size = 64
train_loader = DataLoaderMulti(train_data, (train_label_L1, train_label_L2), batch_size, shuffle=True)
test_loader = DataLoaderMulti(test_data, (test_label_L1, test_label_L2), batch_size, shuffle=False)

print("Create train and test loader done.")
print("Start training...")
optimizer = torch.optim.Adam(P.parameters(), lr=0.001)

for epoch in range(1):
    train_loss = 0.0

    for batch_idx, (data, label_L1, label_L2) in enumerate(train_loader):
        optimizer.zero_grad()

        x = ltn.Variable("x", data)
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 0])
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 1])
        x_Recon = ltn.Variable("x_Recon", data[label_L1 == 2])
        x_ARP_Spoofing = ltn.Variable("x_ARP_Spoofing", data[label_L1 == 3])
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 4])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 5])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 6])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 7])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 8])
        x_benign = ltn.Variable("x_benign", data[label_L2 == 9])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 10])
        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 11])
        x_arp_spoofing = ltn.Variable("x_arp_spoofing", data[label_L2 == 12])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_MQTT, l_MQTT),
            (x_Benign, l_Benign),
            (x_Recon, l_Recon),
            (x_ARP_Spoofing, l_ARP_Spoofing),
            (x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
            (x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
            (x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
            (x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
            (x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
            (x_benign, l_benign),
            (x_Recon_OS_Scan, l_Recon_OS_Scan),
            (x_Recon_Port_Scan, l_Recon_Port_Scan),
            (x_arp_spoofing, l_arp_spoofing)
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label)))

        # rules - L1 class exclusive for each other
        valid_forall_expressions.append(Exists(x, Not(And(P(x, l_MQTT), P(x, l_Benign), P(x, l_Recon), P(x, l_ARP_Spoofing)))))
        # rules - L2 exclusive
        valid_forall_expressions.append(Exists(x, Not(And(P(x, l_MQTT_DDoS_Connect_Flood), P(x, l_MQTT_DDoS_Publish_Flood), 
                                                          P(x, l_MQTT_DoS_Connect_Flood), P(x, l_MQTT_DoS_Publish_Flood),
                                                          P(x, l_MQTT_Malformed_Data), P(x, l_benign),
                                                          P(x, l_Recon_OS_Scan), P(x, l_Recon_Port_Scan),
                                                          P(x, l_arp_spoofing)))))
        # rules - hierarchy
        valid_forall_expressions.append(Exists(x, And(P(x, l_Benign), P(x, l_benign))))
        valid_forall_expressions.append(Exists(x, And(P(x, l_ARP_Spoofing), P(x, l_arp_spoofing))))

        
        
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
    
        logging.info(f"epoch {epoch} | loss {train_loss:.4f} | Train Sat {train_sat:.3f} | Test Sat {test_sat:.3f}")
        logging.info(f"{' ' * 10}Train Acc L1 {train_acc[0]:.3f} | Train Acc L2 {train_acc[1]:.3f} | "
                     f"Test Acc L1 {test_acc[0]:.3f} | Test Acc L2 {test_acc[1]:.3f}")
    # if epoch % 10 == 0:
    #     print(f"Test Sat Phi 1 {compute_sat_level_phi(test_loader, phi1):.3f} | Test Sat Phi 2 {compute_sat_level_phi(test_loader, phi2):.3f} | "
    #         #   f"Test Sat Phi 3 {compute_sat_level_phi(test_loader, phi3):.3f} | Test Sat Phi 4 {compute_sat_level_phi(test_loader, phi4):.3f}"
    #           )
    #     logging.info(f"Test Sat Phi 1 {compute_sat_level_phi(test_loader, phi1):.3f} | Test Sat Phi 2 {compute_sat_level_phi(test_loader, phi2):.3f} | "
    #                 #  f"Test Sat Phi 3 {compute_sat_level_phi(test_loader, phi3):.3f} | Test Sat Phi 4 {compute_sat_level_phi(test_loader, phi4):.3f}"
    #                  )

#####################Evaluation#################################
def compute_weighted_accuracy(loader):
    total_weighted_score = 0.0
    total_instances = 0

    for data, label_L1, label_L2 in loader:
        # Get predictions from the MLP model
        predictions = mlp(data).detach().cpu().numpy()

        # Predicted and true classes for Label_L1
        pred_L1 = np.argmax(predictions[:, 0:4], axis=-1)
        true_L1 = label_L1.cpu().numpy()

        # Predicted and true classes for Label_L2
        pred_L2 = np.argmax(predictions[:, 4:], axis=-1) + 4  # Shift range to match [4, 12]
        true_L2 = label_L2.cpu().numpy()

        # Compute instance-wise weights
        for p_L1, t_L1, p_L2, t_L2 in zip(pred_L1, true_L1, pred_L2, true_L2):
            if p_L1 == t_L1 and p_L2 == t_L2:  # Case 1: Both correct
                weight = 1
            elif p_L1 == t_L1 and p_L2 != t_L2:  # Case 2: L1 correct, L2 incorrect
                weight = 0.5
            elif p_L1 != t_L1 and p_L2 == t_L2:  # Case 3: L2 correct, L1 incorrect
                weight = 0.5
            else:  # Case 4: Both incorrect
                weight = 0

            # Add weight to total score
            total_weighted_score += weight

        # Count total instances
        total_instances += len(pred_L1)

    # Compute weighted accuracy
    weighted_accuracy = total_weighted_score / total_instances

    return weighted_accuracy

class_names_L1 = ["MQTT", "Benign", "Recon", "ARP_Spoofing"]
class_names_L2 = [
    "MQTT-DDoS-Connect_Flood", 
    "MQTT-DDoS-Publish_Flood", 
    "MQTT-DoS-Connect_Flood", 
    "MQTT-DoS-Publish_Flood", 
    "MQTT-Malformed_Data",
    "benign",
    'Recon-Port_Scan',
    'Recon-OS_Scan',
    'arp_spoofing'
]

report_L1, report_L2 = compute_metrics(test_loader, mlp, class_names_L1, class_names_L2)
logging.info(f"\n {report_L1}")
logging.info(f"\n {report_L2} \n")
plot_combined_confusion_matrices(test_loader, mlp, class_names_L1, class_names_L2, filename='4-9_LTN_2_s')


weighted_accuracy = compute_weighted_accuracy(test_loader)
print(f"Weighted Accuracy: {weighted_accuracy:.3f}")
logging.info(f"Weighted Accuracy: {weighted_accuracy:.3f} \n\n")

#####################LIME#################################
from lime.lime_tabular import LimeTabularExplainer

# Prepare training data and feature names
X_train = train_data.cpu().numpy()
class_names_all = class_names_L1 + class_names_L2  # Replace with actual feature names

# Create LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=class_names_all,
    class_names=class_names_L2,  # Class names for L2 predictions
    mode="classification"
)

# Select a single instance for explanation
test_instance = test_data[0].cpu().numpy()  # Example instance
true_label_L2 = test_label_L2[0].item()  # True label for L2

# Define prediction function
def predict_fn(data):
    data_tensor = torch.tensor(data).float().to(device)  # Convert to tensor and move to device
    logits = mlp(data_tensor).detach().cpu().numpy()  # Get logits from MLP
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
    return probabilities

# Generate explanation for the instance
explanation = explainer.explain_instance(
    data_row=test_instance,
    predict_fn=predict_fn
)

# Visualize the explanation
explanation.save_to_file("lime_explanation.html")  # Save as HTML for external viewing




