import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from utils import MLP, LogitsToPredicate, DataLoader
import custom_fuzzy_ops


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
    # for data, labels in loader:
    #     x_A = ltn.Variable("x_A", data[labels == 0])
    #     x_B = ltn.Variable("x_B", data[labels == 1])
    #     x_C = ltn.Variable("x_C", data[labels == 2])
    #     x_D = ltn.Variable("x_D", data[labels == 3])
    #     x_E = ltn.Variable("x_E", data[labels == 4])
    #     x_F = ltn.Variable("x_F", data[labels == 5])
    #     ###########################################################
    #     valid_forall_expressions = []
    #     variables_labels = [(x_A, l_A),
    #                         (x_B, l_B),
    #                         (x_C, l_C),
    #                         (x_D, l_D),
    #                         (x_E, l_E),
    #                         (x_F, l_F),
    #                         ]
    #     for variable, label in variables_labels:
    #         if variable.value.size(0) != 0:
    #             valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
    #     mean_sat += SatAgg(*valid_forall_expressions)
    #     ##############################################################
    # mean_sat /= len(loader)
    # return mean_sat




# 加载数据集
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数，适用于19个类别的场景 ('label_L1' remains in the train_data and test_data)
train_label_L1, train_label_L2 = train_data("label_L1"), train_data.pop("label_L2")
test_label_L1, test_label_L2 = test_data("label_L1"), test_data.pop("label_L2")

label_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5,
                 "MQTT-DDoS-Connect_Flood": 6, "MQTT-DDoS-Publish_Flood": 7, "MQTT-DoS-Connect_Flood": 8,
                 "MQTT-DoS-Publish_Flood": 9, "MQTT-Malformed_Data": 10,
                 "Recon-Port_Scan": 11, "Recon-OS_Scan": 12, "Recon-VulScan": 13, "Recon-Ping_Sweep": 14,
                 "TCP_IP-DDoS-TCP": 15, "TCP_IP-DDoS-ICMP": 16,  "TCP_IP-DDoS-SYN": 17, "TCP_IP-DDoS-UDP": 18,
                 "TCP_IP-DoS-TCP": 19, "TCP_IP-DoS-ICMP": 20, "TCP_IP-DoS-SYN": 21, "TCP_IP-DoS-UDP": 22,
                 "benign": 23,
                 "arp_spoofing": 24}
train_label_L1, train_label_L2 = train_label_L1.map(label_mapping), train_label_L2.map(label_mapping)
test_label_L1, test_label_L2 = test_label_L1.map(label_mapping), test_label_L2.map(label_mapping)

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
l_Benign = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_ARP_Spoofing = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

l_MQTT_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Port_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_OS_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_VulScan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Ping_Sweep = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
l_TCP_IP_DoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
l_TCP_IP_DoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
l_TCP_IP_DoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
l_TCP_IP_DoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
l_benign = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
l_arp_spoofing = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))

# 创建模型实例并移动到设备 
# Notice: Input layer has 46 features
mlp = MLP(layer_sizes=(46, 64, 32, 19)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
P = ltn.Predicate(LogitsToPredicate(mlp))

# define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(custom_fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

print("LTN setting done.")


#####################Training#################################
# create train and test loader
batch_size = 64
train_loader = DataLoader(train_data, train_label_L2, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label_L2, batch_size, shuffle=False)

# Learning
optimizer = torch.optim.Adam(P.parameters(), lr=0.0001)

for epoch in range(100):
    train_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()