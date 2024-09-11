import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from utils import MLP, LogitsToPredicate, DataLoader
import custom_fuzzy_ops as custom_fuzzy_ops


# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    mean_accuracy = 0.0
    for data, labels in loader:
        # 确保数据在正确的设备上
        data, labels = data.to(device), labels.to(device)
        predictions = mlp(data).detach()
        predictions = predictions.cpu().numpy()  # 先移动到CPU，然后转换为NumPy数组
        predictions = np.argmax(predictions, axis=1)
        mean_accuracy += accuracy_score(labels.cpu().numpy(), predictions)  # 同样，确保labels也在CPU上

    return mean_accuracy / len(loader)


# define metrics for evaluation of the model
def compute_metrics(loader, model):
    all_labels = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            # 进行预测
            predictions = model(data).detach()
            predictions = predictions.cpu().numpy()
            predicted_classes = np.argmax(predictions, axis=1)

            # 收集所有的标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes)

    # 计算 precision, recall 和 f1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        # average='macro',
        zero_division=0  # 防止由于没有预测样本导致的未定义行为
    )

    return precision, recall, f1_score


# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, label_L2 in loader:
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 0])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 1])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 2])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 3])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 4])
        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 5])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 6])
        x_Recon_VulScan = ltn.Variable("x_Recon_VulScan", data[label_L2 == 7])
        x_Recon_Ping_Sweep = ltn.Variable("x_Recon_Ping_Sweep", data[label_L2 == 8])
        x_TCP_IP_DDoS_TCP = ltn.Variable("x_TCP_IP_DDoS_TCP", data[label_L2 == 9])
        x_TCP_IP_DDoS_ICMP = ltn.Variable("x_TCP_IP_DDoS_ICMP", data[label_L2 == 10])
        x_TCP_IP_DDoS_SYN = ltn.Variable("x_TCP_IP_DDoS_SYN", data[label_L2 == 11])
        x_TCP_IP_DDoS_UDP = ltn.Variable("x_TCP_IP_DDoS_UDP", data[label_L2 == 12])
        x_TCP_IP_DoS_TCP = ltn.Variable("x_TCP_IP_DoS_TCP", data[label_L2 == 13])
        x_TCP_IP_DoS_ICMP = ltn.Variable("x_TCP_IP_DoS_ICMP", data[label_L2 == 14])
        x_TCP_IP_DoS_SYN = ltn.Variable("x_TCP_IP_DoS_SYN", data[label_L2 == 15])
        x_TCP_IP_DoS_UDP = ltn.Variable("x_TCP_IP_DoS_UDP", data[label_L2 == 16])
        x_benign = ltn.Variable("x_benign", data[label_L2 == 17])
        x_arp_spoofing = ltn.Variable("x_arp_spoofing", data[label_L2 == 18])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
            (x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
            (x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
            (x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
            (x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
            (x_Recon_Port_Scan, l_Recon_Port_Scan),
            (x_Recon_OS_Scan, l_Recon_OS_Scan),
            (x_Recon_VulScan, l_Recon_VulScan),
            (x_Recon_Ping_Sweep, l_Recon_Ping_Sweep),
            (x_TCP_IP_DDoS_TCP, l_TCP_IP_DDoS_TCP),
            (x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDoS_ICMP),
            (x_TCP_IP_DDoS_SYN, l_TCP_IP_DDoS_SYN),
            (x_TCP_IP_DDoS_UDP, l_TCP_IP_DDoS_UDP),
            (x_TCP_IP_DoS_TCP, l_TCP_IP_DoS_TCP),
            (x_TCP_IP_DoS_ICMP, l_TCP_IP_DoS_ICMP),
            (x_TCP_IP_DoS_SYN, l_TCP_IP_DoS_SYN),
            (x_TCP_IP_DoS_UDP, l_TCP_IP_DoS_UDP),
            (x_benign, l_benign),
            (x_arp_spoofing, l_arp_spoofing),
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
        mean_sat += SatAgg(*valid_forall_expressions)
    # In the loop: mean_sat accumulates the satisfaction levels for all the logical rules across the batches.
    # After the loop: mean_sat becomes the average satisfaction level of the logical rules over the entire dataset.
    mean_sat /= len(loader)
    return mean_sat




# 加载数据集
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数，适用于19个类别的场景 ('label_L1' remains in the train_data and test_data)
label_L1_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 0, "MQTT-DDoS-Publish_Flood": 1, "MQTT-DoS-Connect_Flood": 2, "MQTT-DoS-Publish_Flood": 3, "MQTT-Malformed_Data": 4,
                 "Recon-Port_Scan": 5, "Recon-OS_Scan": 6, "Recon-VulScan": 7, "Recon-Ping_Sweep": 8,
                 "TCP_IP-DDoS-TCP": 9, "TCP_IP-DDoS-ICMP": 10,  "TCP_IP-DDoS-SYN": 11, "TCP_IP-DDoS-UDP": 12,
                 "TCP_IP-DoS-TCP": 13, "TCP_IP-DoS-ICMP": 14, "TCP_IP-DoS-SYN": 15, "TCP_IP-DoS-UDP": 16,
                 "benign": 17, "arp_spoofing": 18}
# 应用标签映射并确保存回列
train_data["label_L1"] = train_data["label_L1"].map(label_L1_mapping)
train_data["label_L2"] = train_data["label_L2"].map(label_L2_mapping)
test_data["label_L1"] = test_data["label_L1"].map(label_L1_mapping)
test_data["label_L2"] = test_data["label_L2"].map(label_L2_mapping)

# 移除 'label_L2' 并保留 'label_L1'
train_label_L1, train_label_L2 = train_data["label_L1"], train_data.pop("label_L2")
test_label_L1, test_label_L2 = test_data["label_L1"], test_data.pop("label_L2")


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
# 并移动到设备
# we define the constants with 25 classes, including 'benign' and 'arp_spoofing'

# l_Benign = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
# l_MQTT = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
# l_Recon = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
# l_ARP_Spoofing = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
# l_TCP_IP_DDOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
# l_TCP_IP_DOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

# one-hot encoded vector of length 19
l_MQTT_DDoS_Connect_Flood = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Port_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_OS_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_VulScan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Ping_Sweep = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
l_TCP_IP_DoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
l_TCP_IP_DoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
l_TCP_IP_DoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
l_benign = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
l_arp_spoofing = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

# 创建模型实例并移动到设备 
# Notice: Input layer has 46 features
mlp = MLP(layer_sizes=(46, 64, 32, 19)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
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
# create train and test loader
batch_size = 64
train_loader = DataLoader(train_data, train_label_L2, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label_L2, batch_size, shuffle=False)

print("Create train and test loader done.")

print("Start training...")
optimizer = torch.optim.Adam(P.parameters(), lr=0.0001)

for epoch in range(5):
    train_loss = 0.0

    for batch_idx, (data, label_L2) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x = ltn.Variable("x", data)
        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 0])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 1])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 2])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 3])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 4])
        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 5])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 6])
        x_Recon_VulScan = ltn.Variable("x_Recon_VulScan", data[label_L2 == 7])
        x_Recon_Ping_Sweep = ltn.Variable("x_Recon_Ping_Sweep", data[label_L2 == 8])
        x_TCP_IP_DDoS_TCP = ltn.Variable("x_TCP_IP_DDoS_TCP", data[label_L2 == 9])
        x_TCP_IP_DDoS_ICMP = ltn.Variable("x_TCP_IP_DDoS_ICMP", data[label_L2 == 10])
        x_TCP_IP_DDoS_SYN = ltn.Variable("x_TCP_IP_DDoS_SYN", data[label_L2 == 11])
        x_TCP_IP_DDoS_UDP = ltn.Variable("x_TCP_IP_DDoS_UDP", data[label_L2 == 12])
        x_TCP_IP_DoS_TCP = ltn.Variable("x_TCP_IP_DoS_TCP", data[label_L2 == 13])
        x_TCP_IP_DoS_ICMP = ltn.Variable("x_TCP_IP_DoS_ICMP", data[label_L2 == 14])
        x_TCP_IP_DoS_SYN = ltn.Variable("x_TCP_IP_DoS_SYN", data[label_L2 == 15])
        x_TCP_IP_DoS_UDP = ltn.Variable("x_TCP_IP_DoS_UDP", data[label_L2 == 16])
        x_benign = ltn.Variable("x_benign", data[label_L2 == 17])
        x_arp_spoofing = ltn.Variable("x_arp_spoofing", data[label_L2 == 18])

        # rules - single class exclusive
        valid_forall_expressions = []

        variables_labels = [
            (x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
            (x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
            (x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
            (x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
            (x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
            (x_Recon_Port_Scan, l_Recon_Port_Scan),
            (x_Recon_OS_Scan, l_Recon_OS_Scan),
            (x_Recon_VulScan, l_Recon_VulScan),
            (x_Recon_Ping_Sweep, l_Recon_Ping_Sweep),
            (x_TCP_IP_DDoS_TCP, l_TCP_IP_DDoS_TCP),
            (x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDoS_ICMP),
            (x_TCP_IP_DDoS_SYN, l_TCP_IP_DDoS_SYN),
            (x_TCP_IP_DDoS_UDP, l_TCP_IP_DDoS_UDP),
            (x_TCP_IP_DoS_TCP, l_TCP_IP_DoS_TCP),
            (x_TCP_IP_DoS_ICMP, l_TCP_IP_DoS_ICMP),
            (x_TCP_IP_DoS_SYN, l_TCP_IP_DoS_SYN),
            (x_TCP_IP_DoS_UDP, l_TCP_IP_DoS_UDP),
            (x_benign, l_benign),
            (x_arp_spoofing, l_arp_spoofing),
        ]
        
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))

        sat_agg = SatAgg(*valid_forall_expressions) # the satisfaction level over the current batch
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # print metrics
    if epoch % 1 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f"
              % (epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
                 compute_accuracy(train_loader), compute_accuracy(test_loader)))
        # Evaluate
        precision, recall, f1 = compute_metrics(test_loader, mlp)
        print(f"Macro Recall: {recall.mean():.4f}, Macro Precision: {precision.mean():.4f}, Macro F1-Score: {f1.mean():.4f}")