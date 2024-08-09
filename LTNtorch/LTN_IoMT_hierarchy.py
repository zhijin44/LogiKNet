import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
import custom_fuzzy_ops

# 加载数据集
processed_train_file = '../CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/reduced_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数，适用于19个类别的场景
train_label_L1, train_label_L2 = train_data.pop("label_L1"), train_data.pop("label_L2")
test_label_L1, test_label_L2 = test_data.pop("label_L1"), test_data.pop("label_L2")

label_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5,
                 "MQTT-DDoS-Connect_Flood": 6, "MQTT-DDoS-Publish_Flood": 7, "MQTT-DoS-Connect_Flood": 8,
                 "MQTT-DoS-Publish_Flood": 9, "MQTT-Malformed_Data": 10,
                 "Recon-Port_Scan": 11, "Recon-OS_Scan": 12, "Recon-VulScan": 13, "Recon-Ping_Sweep": 14,
                 "TCP_IP-DDoS-TCP": 15, "TCP_IP-DDoS-ICMP": 16,  "TCP_IP-DDoS-SYN": 17, "TCP_IP-DDoS-UDP": 18,
                 "TCP_IP-DoS-TCP": 19, "TCP_IP-DoS-ICMP": 20, "TCP_IP-DoS-SYN": 21, "TCP_IP-DoS-UDP": 22}
train_label_L1, train_label_L2 = train_label_L1.map(label_mapping), train_label_L2.map(label_mapping)
test_label_L1, test_label_L2 = test_label_L1.map(label_mapping), test_label_L2.map(label_mapping)

# 使用 StandardScaler 对数据进行缩放
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# # 打印部分数据以确认
# print(train_data)
# print(train_data_scaled[:5])
#
# # 查看标签映射后的结果
# print(train_label_L1.head()), print(train_label_L2.head())
# print(test_label_L1.head()), print(test_label_L2.head())

print("Data processing and scaling done.")

# 将缩放后的数据和标签转换为Tensor
# 定义设备并移动数据和标签到设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torch.tensor(train_data_scaled).float().to(device)
test_data = torch.tensor(test_data_scaled).float().to(device)
train_label_L1 = torch.tensor(train_label_L1.to_numpy()).long().to(device)
train_label_L2 = torch.tensor(train_label_L2.to_numpy()).long().to(device)
test_label_L1 = torch.tensor(test_label_L1.to_numpy()).long().to(device)
test_label_L2 = torch.tensor(test_label_L2.to_numpy()).long().to(device)

#####################Setting#################################
# we define the constants
# 并移动到设备
l_Benign = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_ARP_Spoofing = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DOS = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_MQTT_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Port_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_OS_Scan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_VulScan = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
l_Recon_Ping_Sweep = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
l_TCP_IP_DDoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
l_TCP_IP_DDoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
l_TCP_IP_DoS_TCP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
l_TCP_IP_DoS_ICMP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
l_TCP_IP_DoS_SYN = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
l_TCP_IP_DoS_UDP = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))


# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes=(45, 16, 16, 8, 23)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, x, training=False):
        """
        Method which defines the forward phase of the neural network for our multi class classification task.
        In particular, it returns the logits for the classes given an input example.

        :param x: the features of the example
        :param training: whether the network is in training mode (dropout applied) or validation mode (dropout not applied)
        :return: logits for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
        logits = self.linear_layers[-1](x)
        return logits


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """

    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, l, training=False):
        logits = self.logits_model(x, training=training)
        probs = self.sigmoid(logits)
        out = torch.sum(probs * l, dim=1)
        return out


mlp = MLP().to(device)
P = ltn.Predicate(LogitsToPredicate(mlp))

# we define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(custom_fuzzy_ops.AndProd())
# And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

print("LTN setting done.")


#####################utils#####################################
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.label_L1 = labels[0]
        self.label_L2 = labels[1]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            label_L1 = self.label_L1[idxlist[start_idx:end_idx]]
            label_L2 = self.label_L2[idxlist[start_idx:end_idx]]

            yield data, label_L1, label_L2


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, label_L1, label_L2 in loader:
        x = ltn.Variable("x", data)
        # 根据label_mapping创建变量
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 0])
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 1])
        x_Recon = ltn.Variable("x_Recon", data[label_L1 == 2])
        x_ARP_Spoofing = ltn.Variable("x_ARP_Spoofing", data[label_L1 == 3])
        x_TCP_IP_DDOS = ltn.Variable("x_TCP_IP_DDOS", data[label_L1 == 4])
        x_TCP_IP_DOS = ltn.Variable("x_TCP_IP_DOS", data[label_L1 == 5])

        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 6])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 7])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 8])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 9])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 10])

        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 11])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 12])
        x_Recon_VulScan = ltn.Variable("x_Recon_VulScan", data[label_L2 == 13])
        x_Recon_Ping_Sweep = ltn.Variable("x_Recon_Ping_Sweep", data[label_L2 == 14])

        x_TCP_IP_DDoS_TCP = ltn.Variable("x_TCP_IP_DDoS_TCP", data[label_L2 == 15])
        x_TCP_IP_DDoS_ICMP = ltn.Variable("x_TCP_IP_DDoS_ICMP", data[label_L2 == 16])
        x_TCP_IP_DDoS_SYN = ltn.Variable("x_TCP_IP_DDoS_SYN", data[label_L2 == 17])
        x_TCP_IP_DDoS_UDP = ltn.Variable("x_TCP_IP_DDoS_UDP", data[label_L2 == 18])

        x_TCP_IP_DoS_TCP = ltn.Variable("x_TCP_IP_DoS_TCP", data[label_L2 == 19])
        x_TCP_IP_DoS_ICMP = ltn.Variable("x_TCP_IP_DoS_ICMP", data[label_L2 == 20])
        x_TCP_IP_DoS_SYN = ltn.Variable("x_TCP_IP_DoS_SYN", data[label_L2 == 21])
        x_TCP_IP_DoS_UDP = ltn.Variable("x_TCP_IP_DoS_UDP", data[label_L2 == 22])

        # 创建有效的forall表达式列表
        valid_forall_expressions = []
        if x_Benign.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Benign, P(x_Benign, l_Benign)))
        if x_MQTT.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_MQTT, P(x_MQTT, l_MQTT)))
        if x_Recon.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon, P(x_Recon, l_Recon)))
        if x_ARP_Spoofing.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_ARP_Spoofing, P(x_ARP_Spoofing, l_ARP_Spoofing)))
        if x_TCP_IP_DDOS.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDOS, P(x_TCP_IP_DDOS, l_TCP_IP_DDOS)))
        if x_TCP_IP_DOS.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DOS, P(x_TCP_IP_DOS, l_TCP_IP_DOS)))

        if x_MQTT_DDoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DDoS_Connect_Flood, And(P(x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
                                                      P(x_MQTT_DDoS_Connect_Flood, l_MQTT))))
        if x_MQTT_DDoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DDoS_Publish_Flood, And(P(x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
                                                      P(x_MQTT_DDoS_Publish_Flood, l_MQTT))))
        if x_MQTT_DoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DoS_Connect_Flood, And(P(x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
                                                     P(x_MQTT_DoS_Connect_Flood, l_MQTT))))
        if x_MQTT_DoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DoS_Publish_Flood, And(P(x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
                                                     P(x_MQTT_DoS_Publish_Flood, l_MQTT))))
        if x_MQTT_Malformed_Data.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_Malformed_Data, And(P(x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
                                                  P(x_MQTT_Malformed_Data, l_MQTT))))

        if x_Recon_Port_Scan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_Port_Scan, And(P(x_Recon_Port_Scan, l_Recon_Port_Scan),
                                                                          P(x_Recon_Port_Scan, l_Recon))))
        if x_Recon_OS_Scan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_OS_Scan, And(P(x_Recon_OS_Scan, l_Recon_OS_Scan),
                                                                        P(x_Recon_OS_Scan, l_Recon))))
        if x_Recon_VulScan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_VulScan, And(P(x_Recon_VulScan, l_Recon_VulScan),
                                                                        P(x_Recon_VulScan, l_Recon))))
        if x_Recon_Ping_Sweep.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_Ping_Sweep, And(P(x_Recon_Ping_Sweep, l_Recon_Ping_Sweep),
                                                                           P(x_Recon_Ping_Sweep, l_Recon))))

        if x_TCP_IP_DDoS_TCP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_TCP, And(P(x_TCP_IP_DDoS_TCP, l_TCP_IP_DDoS_TCP),
                                                                          P(x_TCP_IP_DDoS_TCP, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_ICMP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_ICMP, And(P(x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDoS_ICMP),
                                                                           P(x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_SYN.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_SYN, And(P(x_TCP_IP_DDoS_SYN, l_TCP_IP_DDoS_SYN),
                                                                          P(x_TCP_IP_DDoS_SYN, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_UDP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_UDP, And(P(x_TCP_IP_DDoS_UDP, l_TCP_IP_DDoS_UDP),
                                                                          P(x_TCP_IP_DDoS_UDP, l_TCP_IP_DDOS))))

        if x_TCP_IP_DoS_TCP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_TCP, And(P(x_TCP_IP_DoS_TCP, l_TCP_IP_DoS_TCP),
                                                                         P(x_TCP_IP_DoS_TCP, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_ICMP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_ICMP, And(P(x_TCP_IP_DoS_ICMP, l_TCP_IP_DoS_ICMP),
                                                                          P(x_TCP_IP_DoS_ICMP, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_SYN.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_SYN, And(P(x_TCP_IP_DoS_SYN, l_TCP_IP_DoS_SYN),
                                                                         P(x_TCP_IP_DoS_SYN, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_UDP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_UDP, And(P(x_TCP_IP_DoS_UDP, l_TCP_IP_DoS_UDP),
                                                                         P(x_TCP_IP_DoS_UDP, l_TCP_IP_DOS))))

        mutual_exclusive_constraints = [
            Forall(x, Not(And(P(x, l_Benign), P(x, l_MQTT), P(x, l_Recon), P(x, l_ARP_Spoofing), P(x, l_TCP_IP_DDOS), P(x, l_TCP_IP_DOS)))),
            Forall(x_MQTT, Not(And(P(x_MQTT, l_MQTT_DDoS_Connect_Flood), P(x_MQTT, l_MQTT_DDoS_Publish_Flood), P(x_MQTT, l_MQTT_DoS_Connect_Flood), 
                                   P(x_MQTT, l_MQTT_DoS_Publish_Flood), P(x_MQTT, l_MQTT_Malformed_Data)))),
            Forall(x_Recon, Not(And(P(x_Recon, l_Recon_OS_Scan), P(x_Recon, l_Recon_Ping_Sweep), P(x_Recon, l_Recon_Port_Scan), 
                                    P(x_Recon, l_Recon_VulScan)))),
            Forall(x_TCP_IP_DDOS, Not(And(P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_ICMP), P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_SYN), 
                                          P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_TCP), P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_UDP)))),
            Forall(x_TCP_IP_DOS, Not(And(P(x_TCP_IP_DOS, l_TCP_IP_DoS_ICMP), P(x_TCP_IP_DOS, l_TCP_IP_DoS_SYN), 
                                         P(x_TCP_IP_DOS, l_TCP_IP_DoS_TCP), P(x_TCP_IP_DOS, l_TCP_IP_DoS_UDP))))
            # Forall(x, Not(And(P(x, l_MQTT_DDoS_Connect_Flood), P(x, l_MQTT_DDoS_Publish_Flood), P(x, l_MQTT_DoS_Connect_Flood), 
            #                   P(x, l_MQTT_DoS_Publish_Flood), P(x, l_MQTT_Malformed_Data)))),
            # Forall(x, Not(And(P(x, l_Recon_OS_Scan), P(x, l_Recon_Ping_Sweep), P(x, l_Recon_Port_Scan), P(x, l_Recon_VulScan)))),
            # Forall(x, Not(And(P(x, l_TCP_IP_DDoS_ICMP), P(x, l_TCP_IP_DDoS_SYN), P(x, l_TCP_IP_DDoS_TCP), P(x, l_TCP_IP_DDoS_UDP)))),
            # Forall(x, Not(And(P(x, l_TCP_IP_DoS_ICMP), P(x, l_TCP_IP_DoS_SYN), P(x, l_TCP_IP_DoS_TCP), P(x, l_TCP_IP_DoS_UDP))))
        ]
        valid_forall_expressions.extend(mutual_exclusive_constraints)

        mean_sat += SatAgg(*valid_forall_expressions)
    mean_sat /= len(loader)
    return mean_sat


# it computes the overall accuracy of the predictions of the trained model using the given data loader (train or test)
def compute_accuracy(loader, threshold=0.5):
    mean_accuracy = 0.0
    for data, label_L1, label_L2 in loader:
        # 确保数据在正确的设备上
        data, label_L1, label_L2 = data.to(device), label_L1.to(device), label_L2.to(device)
        predictions = mlp(data).detach().numpy()
        label_Benign = (label_L1 == 0)
        label_MQTT = (label_L1 == 1)
        label_Recon = (label_L1 == 2)
        label_ARP_Spoofing = (label_L1 == 3)
        label_TCP_IP_DDOS = (label_L1 == 4)
        label_TCP_IP_DOS = (label_L1 == 5)
        label_MQTT_DDoS_Connect_Flood = (label_L2 == 6)
        label_MQTT_DDoS_Publish_Flood = (label_L2 == 7)
        label_MQTT_DoS_Connect_Flood = (label_L2 == 8)
        label_MQTT_DoS_Publish_Flood = (label_L2 == 9)
        label_MQTT_Malformed_Data = (label_L2 == 10)
        label_Recon_Port_Scan = (label_L2 == 11)
        label_Recon_OS_Scan = (label_L2 == 12)
        label_Recon_VulScan = (label_L2 == 13)
        label_Recon_Ping_Sweep = (label_L2 == 14)
        label_TCP_IP_DDoS_TCP = (label_L2 == 15)
        label_TCP_IP_DDoS_ICMP = (label_L2 == 16)
        label_TCP_IP_DDoS_SYN = (label_L2 == 17)
        label_TCP_IP_DDoS_UDP = (label_L2 == 18)
        label_TCP_IP_DoS_TCP = (label_L2 == 19)
        label_TCP_IP_DoS_ICMP = (label_L2 == 20)
        label_TCP_IP_DoS_SYN = (label_L2 == 21)
        label_TCP_IP_DoS_UDP = (label_L2 == 22)

        onehot = (np.stack(
            [label_Benign, label_MQTT, label_Recon, label_ARP_Spoofing, label_TCP_IP_DDOS, label_TCP_IP_DOS,
             label_MQTT_DDoS_Connect_Flood, label_MQTT_DDoS_Publish_Flood, label_MQTT_DoS_Connect_Flood,
             label_MQTT_DoS_Publish_Flood, label_MQTT_Malformed_Data,
             label_Recon_Port_Scan, label_Recon_OS_Scan, label_Recon_VulScan, label_Recon_Ping_Sweep,
             label_TCP_IP_DDoS_TCP, label_TCP_IP_DDoS_ICMP, label_TCP_IP_DDoS_SYN, label_TCP_IP_DDoS_UDP,
             label_TCP_IP_DoS_TCP, label_TCP_IP_DoS_ICMP, label_TCP_IP_DoS_SYN, label_TCP_IP_DoS_UDP], axis=-1).astype(
            np.int32))

        predictions = predictions > threshold
        predictions = predictions.astype(np.int32)
        nonzero = np.count_nonzero(onehot - predictions, axis=-1).astype(np.float32)
        multilabel_hamming_loss = nonzero / predictions.shape[-1]
        mean_accuracy += np.mean(1 - multilabel_hamming_loss)

    return mean_accuracy / len(loader)


# create train and test loader
train_loader = DataLoader(train_data, (train_label_L1, train_label_L2), 512, shuffle=True)
test_loader = DataLoader(test_data, (test_label_L1, test_label_L2), 512, shuffle=True)
print("Create train and test loader done.")


#####################learning#####################################
# we learn our LTN in the multi-class multi-label classification task using the satisfaction of the knowledge base as
# an objective. In other words, we want to learn the parameters θ of binary predicate P in such a way the three
# axioms in the knowledge base are maximally satisfied.
optimizer = torch.optim.Adam(P.parameters(), lr=0.001)
print("Start training...")

for epoch in range(5):
    train_loss = 0.0
    for batch_idx, (data, label_L1, label_L2) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x = ltn.Variable("x", data)

        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 0])
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 1])
        x_Recon = ltn.Variable("x_Recon", data[label_L1 == 2])
        x_ARP_Spoofing = ltn.Variable("x_ARP_Spoofing", data[label_L1 == 3])
        x_TCP_IP_DDOS = ltn.Variable("x_TCP_IP_DDOS", data[label_L1 == 4])
        x_TCP_IP_DOS = ltn.Variable("x_TCP_IP_DOS", data[label_L1 == 5])

        x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[label_L2 == 6])
        x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[label_L2 == 7])
        x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[label_L2 == 8])
        x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[label_L2 == 9])
        x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[label_L2 == 10])

        x_Recon_Port_Scan = ltn.Variable("x_Recon_Port_Scan", data[label_L2 == 11])
        x_Recon_OS_Scan = ltn.Variable("x_Recon_OS_Scan", data[label_L2 == 12])
        x_Recon_VulScan = ltn.Variable("x_Recon_VulScan", data[label_L2 == 13])
        x_Recon_Ping_Sweep = ltn.Variable("x_Recon_Ping_Sweep", data[label_L2 == 14])

        x_TCP_IP_DDoS_TCP = ltn.Variable("x_TCP_IP_DDoS_TCP", data[label_L2 == 15])
        x_TCP_IP_DDoS_ICMP = ltn.Variable("x_TCP_IP_DDoS_ICMP", data[label_L2 == 16])
        x_TCP_IP_DDoS_SYN = ltn.Variable("x_TCP_IP_DDoS_SYN", data[label_L2 == 17])
        x_TCP_IP_DDoS_UDP = ltn.Variable("x_TCP_IP_DDoS_UDP", data[label_L2 == 18])

        x_TCP_IP_DoS_TCP = ltn.Variable("x_TCP_IP_DoS_TCP", data[label_L2 == 19])
        x_TCP_IP_DoS_ICMP = ltn.Variable("x_TCP_IP_DoS_ICMP", data[label_L2 == 20])
        x_TCP_IP_DoS_SYN = ltn.Variable("x_TCP_IP_DoS_SYN", data[label_L2 == 21])
        x_TCP_IP_DoS_UDP = ltn.Variable("x_TCP_IP_DoS_UDP", data[label_L2 == 22])
        ##############################################################################
        valid_forall_expressions = []
        if x_Benign.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Benign, P(x_Benign, l_Benign)))
        if x_MQTT.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_MQTT, P(x_MQTT, l_MQTT)))
        if x_Recon.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon, P(x_Recon, l_Recon)))
        if x_ARP_Spoofing.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_ARP_Spoofing, P(x_ARP_Spoofing, l_ARP_Spoofing)))
        if x_TCP_IP_DDOS.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDOS, P(x_TCP_IP_DDOS, l_TCP_IP_DDOS)))
        if x_TCP_IP_DOS.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DOS, P(x_TCP_IP_DOS, l_TCP_IP_DOS)))

        if x_MQTT_DDoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DDoS_Connect_Flood, And(P(x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood),
                                                      P(x_MQTT_DDoS_Connect_Flood, l_MQTT))))
        if x_MQTT_DDoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DDoS_Publish_Flood, And(P(x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood),
                                                      P(x_MQTT_DDoS_Publish_Flood, l_MQTT))))
        if x_MQTT_DoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DoS_Connect_Flood, And(P(x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood),
                                                     P(x_MQTT_DoS_Connect_Flood, l_MQTT))))
        if x_MQTT_DoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_DoS_Publish_Flood, And(P(x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood),
                                                     P(x_MQTT_DoS_Publish_Flood, l_MQTT))))
        if x_MQTT_Malformed_Data.value.size(0) != 0:
            valid_forall_expressions.append(
                Forall(x_MQTT_Malformed_Data, And(P(x_MQTT_Malformed_Data, l_MQTT_Malformed_Data),
                                                  P(x_MQTT_Malformed_Data, l_MQTT))))

        if x_Recon_Port_Scan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_Port_Scan, And(P(x_Recon_Port_Scan, l_Recon_Port_Scan),
                                                                          P(x_Recon_Port_Scan, l_Recon))))
        if x_Recon_OS_Scan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_OS_Scan, And(P(x_Recon_OS_Scan, l_Recon_OS_Scan),
                                                                        P(x_Recon_OS_Scan, l_Recon))))
        if x_Recon_VulScan.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_VulScan, And(P(x_Recon_VulScan, l_Recon_VulScan),
                                                                        P(x_Recon_VulScan, l_Recon))))
        if x_Recon_Ping_Sweep.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Recon_Ping_Sweep, And(P(x_Recon_Ping_Sweep, l_Recon_Ping_Sweep),
                                                                           P(x_Recon_Ping_Sweep, l_Recon))))

        if x_TCP_IP_DDoS_TCP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_TCP, And(P(x_TCP_IP_DDoS_TCP, l_TCP_IP_DDoS_TCP),
                                                                          P(x_TCP_IP_DDoS_TCP, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_ICMP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_ICMP, And(P(x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDoS_ICMP),
                                                                           P(x_TCP_IP_DDoS_ICMP, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_SYN.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_SYN, And(P(x_TCP_IP_DDoS_SYN, l_TCP_IP_DDoS_SYN),
                                                                          P(x_TCP_IP_DDoS_SYN, l_TCP_IP_DDOS))))
        if x_TCP_IP_DDoS_UDP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DDoS_UDP, And(P(x_TCP_IP_DDoS_UDP, l_TCP_IP_DDoS_UDP),
                                                                          P(x_TCP_IP_DDoS_UDP, l_TCP_IP_DDOS))))

        if x_TCP_IP_DoS_TCP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_TCP, And(P(x_TCP_IP_DoS_TCP, l_TCP_IP_DoS_TCP),
                                                                         P(x_TCP_IP_DoS_TCP, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_ICMP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_ICMP, And(P(x_TCP_IP_DoS_ICMP, l_TCP_IP_DoS_ICMP),
                                                                          P(x_TCP_IP_DoS_ICMP, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_SYN.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_SYN, And(P(x_TCP_IP_DoS_SYN, l_TCP_IP_DoS_SYN),
                                                                         P(x_TCP_IP_DoS_SYN, l_TCP_IP_DOS))))
        if x_TCP_IP_DoS_UDP.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_TCP_IP_DoS_UDP, And(P(x_TCP_IP_DoS_UDP, l_TCP_IP_DoS_UDP),
                                                                         P(x_TCP_IP_DoS_UDP, l_TCP_IP_DOS))))

        mutual_exclusive_constraints = [
            Forall(x, Not(And(P(x, l_Benign), P(x, l_MQTT), P(x, l_Recon), P(x, l_ARP_Spoofing), P(x, l_TCP_IP_DDOS), P(x, l_TCP_IP_DOS)))),
            Forall(x_MQTT, Not(And(P(x_MQTT, l_MQTT_DDoS_Connect_Flood), P(x_MQTT, l_MQTT_DDoS_Publish_Flood), P(x_MQTT, l_MQTT_DoS_Connect_Flood), 
                                   P(x_MQTT, l_MQTT_DoS_Publish_Flood), P(x_MQTT, l_MQTT_Malformed_Data)))),
            Forall(x_Recon, Not(And(P(x_Recon, l_Recon_OS_Scan), P(x_Recon, l_Recon_Ping_Sweep), P(x_Recon, l_Recon_Port_Scan), 
                                    P(x_Recon, l_Recon_VulScan)))),
            Forall(x_TCP_IP_DDOS, Not(And(P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_ICMP), P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_SYN), 
                                          P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_TCP), P(x_TCP_IP_DDOS, l_TCP_IP_DDoS_UDP)))),
            Forall(x_TCP_IP_DOS, Not(And(P(x_TCP_IP_DOS, l_TCP_IP_DoS_ICMP), P(x_TCP_IP_DOS, l_TCP_IP_DoS_SYN), 
                                         P(x_TCP_IP_DOS, l_TCP_IP_DoS_TCP), P(x_TCP_IP_DOS, l_TCP_IP_DoS_UDP))))
            # Forall(x, Not(And(P(x, l_MQTT_DDoS_Connect_Flood), P(x, l_MQTT_DDoS_Publish_Flood), P(x, l_MQTT_DoS_Connect_Flood), 
            #                   P(x, l_MQTT_DoS_Publish_Flood), P(x, l_MQTT_Malformed_Data)))),
            # Forall(x, Not(And(P(x, l_Recon_OS_Scan), P(x, l_Recon_Ping_Sweep), P(x, l_Recon_Port_Scan), P(x, l_Recon_VulScan)))),
            # Forall(x, Not(And(P(x, l_TCP_IP_DDoS_ICMP), P(x, l_TCP_IP_DDoS_SYN), P(x, l_TCP_IP_DDoS_TCP), P(x, l_TCP_IP_DDoS_UDP)))),
            # Forall(x, Not(And(P(x, l_TCP_IP_DoS_ICMP), P(x, l_TCP_IP_DoS_SYN), P(x, l_TCP_IP_DoS_TCP), P(x, l_TCP_IP_DoS_UDP))))
        ]
        valid_forall_expressions.extend(mutual_exclusive_constraints)

        sat_agg = SatAgg(*valid_forall_expressions)
        #############################################################################
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every epochs of training
    if epoch % 1 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f | " %
              (epoch, train_loss, compute_sat_level(train_loader),
               compute_sat_level(test_loader),
               compute_accuracy(train_loader), compute_accuracy(test_loader),
               ))