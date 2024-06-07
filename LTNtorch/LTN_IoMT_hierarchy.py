import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize

# 加载数据集
processed_train_file = '../CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/reduced_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 将标签映射到整数，适用于19个类别的场景
train_label_L1, train_label_L2 = train_data.pop("label_L1"), train_data.pop("label_L2"),
test_label_L1, test_label_L2 = test_data.pop("label_L1"), test_data.pop("label_L2")

label_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5,
                 "MQTT-DDoS-Connect_Flood": 11, "MQTT-DDoS-Publish_Flood": 12, "MQTT-DoS-Connect_Flood": 13,
                 "MQTT-DoS-Publish_Flood": 14, "MQTT-Malformed_Data": 15,
                 "Recon-Port_Scan": 21, "Recon-OS_Scan": 22, "Recon-VulScan": 23, "Recon-Ping_Sweep": 24,
                 "TCP_IP-DDoS-TCP": 41, "TCP_IP-DDoS-ICMP": 42,  "TCP_IP-DDoS-SYN": 43, "TCP_IP-DDoS-UDP": 44,
                 "TCP_IP-DoS-TCP": 51, "TCP_IP-DoS-ICMP": 52, "TCP_IP-DoS-SYN": 53, "TCP_IP-DoS-UDP": 54}
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
l_Benign = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0]))
l_MQTT = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0]))
l_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0]))
l_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0]))
l_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0]))
l_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0]))
l_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1]))


# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes=(45, 16, 16, 8, 7)):
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
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
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

# label_mapping = {"Benign": 0, "MQTT": 1, "DDoS-Connect_Flood": 2, "DDoS-Publish_Flood": 3,
#                  "DoS-Connect_Flood": 4, "DoS-Publish_Flood": 5, "Malformed_Data": 6}
# l_Benign = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0]))
# l_MQTT = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0]))
# l_DDoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0]))
# l_DDoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0]))
# l_DoS_Connect_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0]))
# l_DoS_Publish_Flood = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0]))
# l_Malformed_Data = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1]))

def compute_sat_level(loader):
    mean_sat = 0
    for data, label_L1, label_L2 in loader:
        x = ltn.Variable("x", data)
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 0])
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 1])
        x_DDoS_Connect_Flood = ltn.Variable("x_DDoS_Connect_Flood", data[label_L2 == 2])
        x_DDoS_Publish_Flood = ltn.Variable("x_DDoS-Publish_Flood", data[label_L2 == 3])
        x_DoS_Connect_Flood = ltn.Variable("x_DoS_Connect_Flood", data[label_L2 == 4])
        x_DoS_Publish_Flood = ltn.Variable("x_DoS_Publish_Flood", data[label_L2 == 5])
        x_Malformed_Data = ltn.Variable("x_Malformed_Data", data[label_L2 == 6])

        # 创建有效的forall表达式列表
        valid_forall_expressions = []
        if x_Benign.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Benign, P(x_Benign, l_Benign)))
        if x_MQTT.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_MQTT, P(x_MQTT, l_MQTT)))
        if x_DDoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DDoS_Connect_Flood, P(x_DDoS_Connect_Flood, l_DDoS_Connect_Flood)))
        if x_DDoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DDoS_Publish_Flood, P(x_DDoS_Publish_Flood, l_DDoS_Publish_Flood)))
        if x_DoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DoS_Connect_Flood, P(x_DoS_Connect_Flood, l_DoS_Connect_Flood)))
        if x_DoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DoS_Publish_Flood, P(x_DoS_Publish_Flood, l_DoS_Publish_Flood)))
        if x_Malformed_Data.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Malformed_Data, P(x_Malformed_Data, l_Malformed_Data)))
        # 添加Benign和MQTT的非逻辑关系
        valid_forall_expressions.append(Forall(x, Not(And(P(x, l_Benign), P(x, l_MQTT)))))
        # 添加label_L2中各标签互斥的语句
        mutual_exclusive_constraints = [
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DDoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DoS_Connect_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_DoS_Connect_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DoS_Connect_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DoS_Connect_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DoS_Publish_Flood), P(x, l_Malformed_Data))))
        ]
        valid_forall_expressions.extend(mutual_exclusive_constraints)

        # mean_sat += SatAgg(
        #     Forall(x_Benign, P(x_Benign, l_Benign)),
        #     Forall(x_MQTT, P(x_MQTT, l_MQTT)),
        #     Forall(x_DDoS_Connect_Flood, P(x_DDoS_Connect_Flood, l_DDoS_Connect_Flood)),
        #     Forall(x_DDoS_Publish_Flood, P(x_DDoS_Publish_Flood, l_DDoS_Publish_Flood)),
        #     Forall(x_DoS_Connect_Flood, P(x_DoS_Connect_Flood, l_DoS_Connect_Flood)),
        #     Forall(x_DoS_Publish_Flood, P(x_DoS_Publish_Flood, l_DoS_Publish_Flood)),
        #     Forall(x_Malformed_Data, P(x_Malformed_Data, l_Malformed_Data)),
        #     Forall(x, Not(And(P(x, l_Benign), P(x, l_MQTT))))
        #     #################此处仍需加一个多重互斥的语句，label_L2中各标签是互斥的###################
        # )
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
        label_DDoS_Connect_Flood = (label_L2 == 2)
        label_DDoS_Publish_Flood = (label_L2 == 3)
        label_DoS_Connect_Flood = (label_L2 == 4)
        label_DoS_Publish_Flood = (label_L2 == 5)
        label_Malformed_Data = (label_L2 == 6)
        onehot = (np.stack(
            [label_Benign, label_MQTT, label_DDoS_Connect_Flood,
             label_DDoS_Publish_Flood, label_DoS_Connect_Flood,
             label_DoS_Publish_Flood, label_Malformed_Data], axis=-1).astype(np.int32))
        predictions = predictions > threshold
        predictions = predictions.astype(np.int32)
        nonzero = np.count_nonzero(onehot - predictions, axis=-1).astype(np.float32)
        multilabel_hamming_loss = nonzero / predictions.shape[-1]
        mean_accuracy += np.mean(1 - multilabel_hamming_loss)

    return mean_accuracy / len(loader)


# create train and test loader
train_loader = DataLoader(train_data, (train_label_L1, train_label_L2), 128, shuffle=True)
test_loader = DataLoader(test_data, (test_label_L1, test_label_L2), 128, shuffle=True)
print("Create train and test loader done.")


#####################learning#####################################
# we learn our LTN in the multi-class multi-label classification task using the satisfaction of the knowledge base as
# an objective. In other words, we want to learn the parameters θ of binary predicate P in such a way the three
# axioms in the knowledge base are maximally satisfied.
optimizer = torch.optim.Adam(P.parameters(), lr=0.001)
print("Start training...")

for epoch in range(50):
    train_loss = 0.0
    for batch_idx, (data, label_L1, label_L2) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x = ltn.Variable("x", data)
        x_Benign = ltn.Variable("x_Benign", data[label_L1 == 0])
        x_MQTT = ltn.Variable("x_MQTT", data[label_L1 == 1])
        x_DDoS_Connect_Flood = ltn.Variable("x_DDoS_Connect_Flood", data[label_L2 == 2])
        x_DDoS_Publish_Flood = ltn.Variable("x_DDoS-Publish_Flood", data[label_L2 == 3])
        x_DoS_Connect_Flood = ltn.Variable("x_DoS_Connect_Flood", data[label_L2 == 4])
        x_DoS_Publish_Flood = ltn.Variable("x_DoS_Publish_Flood", data[label_L2 == 5])
        x_Malformed_Data = ltn.Variable("x_Malformed_Data", data[label_L2 == 6])
        ##############################################################################
        valid_forall_expressions = []
        if x_Benign.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Benign, P(x_Benign, l_Benign)))
        if x_MQTT.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_MQTT, P(x_MQTT, l_MQTT)))
        if x_DDoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DDoS_Connect_Flood, P(x_DDoS_Connect_Flood, l_DDoS_Connect_Flood)))
        if x_DDoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DDoS_Publish_Flood, P(x_DDoS_Publish_Flood, l_DDoS_Publish_Flood)))
        if x_DoS_Connect_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DoS_Connect_Flood, P(x_DoS_Connect_Flood, l_DoS_Connect_Flood)))
        if x_DoS_Publish_Flood.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_DoS_Publish_Flood, P(x_DoS_Publish_Flood, l_DoS_Publish_Flood)))
        if x_Malformed_Data.value.size(0) != 0:
            valid_forall_expressions.append(Forall(x_Malformed_Data, P(x_Malformed_Data, l_Malformed_Data)))
        # 添加Benign和MQTT的非逻辑关系
        valid_forall_expressions.append(Forall(x, Not(And(P(x, l_Benign), P(x, l_MQTT)))))
        # 添加label_L2中各标签互斥的语句
        mutual_exclusive_constraints = [
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DDoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DoS_Connect_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Connect_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_DoS_Connect_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DDoS_Publish_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DoS_Connect_Flood), P(x, l_DoS_Publish_Flood)))),
            Forall(x, Not(And(P(x, l_DoS_Connect_Flood), P(x, l_Malformed_Data)))),
            Forall(x, Not(And(P(x, l_DoS_Publish_Flood), P(x, l_Malformed_Data))))
        ]
        valid_forall_expressions.extend(mutual_exclusive_constraints)

        sat_agg = SatAgg(*valid_forall_expressions)
        #############################################################################
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 20 epochs of training
    if epoch % 1 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f | " %
              (epoch, train_loss, compute_sat_level(train_loader),
               compute_sat_level(test_loader),
               compute_accuracy(train_loader), compute_accuracy(test_loader),
               ))