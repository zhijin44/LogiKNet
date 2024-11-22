import pandas as pd
import numpy as np
import os
import re
import torch
import ltn
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes):
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
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l, training=False):
        logits = self.logits_model(x, training=training)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)  # 计算并返回与给定类标签l对应的概率值
        return out
    

# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data
        self.labels = labels
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
            labels = self.labels[idxlist[start_idx:end_idx]]
            ############################################################
            # Check if any class is missing in the batch
            # present_classes = np.unique(labels.cpu().numpy())
            # all_classes = np.arange(len(label_mapping))  # Adjust based on number of classes
            # missing_classes = set(all_classes) - set(present_classes)
            #
            # if missing_classes:
            #     print(f"Batch {start_idx // self.batch_size} is missing classes {missing_classes}")
            ############################################################
            yield data, labels


def compute_accuracy(loader):
    mean_accuracy_L2 = 0.0
    for data, label_L2 in loader:
        predictions = mlp(data).detach().cpu().numpy()  # Ensure predictions are on CPU for numpy operations
        # Predicted class for Label_L2 
        pred_L2 = np.argmax(predictions, axis=-1) 
        true_L2 = label_L2.cpu().numpy()  # Convert tensor to numpy
        accuracy_L2 = np.mean(pred_L2 == true_L2)
        mean_accuracy_L2 += accuracy_L2
    mean_accuracy_L2 /= len(loader)
    return mean_accuracy_L2


# 定义处理后的数据文件路径
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_small_train.csv'
# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_data.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_data.csv'

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


# 数据预处理
print("Loading processed data...")
training_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)


label_L2_mapping = {
    "MQTT-DDoS-Connect_Flood": 0,
    "MQTT-DDoS-Publish_Flood": 1,
    "MQTT-DoS-Connect_Flood": 2,
    "MQTT-DoS-Publish_Flood": 3,
    "MQTT-Malformed_Data": 4,
    "Benign": 5
}
train_label_L2 = training_data.pop("label_L2").map(label_L2_mapping)
test_label_L2 = test_data.pop("label_L2").map(label_L2_mapping)


print("Scaling data...")
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(training_data[X_columns])
test_data_scaled = scaler.transform(test_data[X_columns])

# 定义设备并移动数据和标签到设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torch.tensor(train_data_scaled).float().to(device)
test_data = torch.tensor(test_data_scaled).float().to(device)
train_label_L2 = torch.tensor(train_label_L2.to_numpy()).long().to(device)
test_label_L2 = torch.tensor(test_label_L2.to_numpy()).long().to(device)

# 创建模型实例并移动到设备 
mlp = MLP(layer_sizes=(45, 32, 32, 6)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
# P = ltn.Predicate(LogitsToPredicate(mlp))
# Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
# SatAgg = ltn.fuzzy_ops.SatAgg()


# create train and test loader (train_sex_labels, train_color_labels)
batch_size = 64
train_loader = DataLoader(train_data, train_label_L2, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label_L2, batch_size, shuffle=False)

print("Training model...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
for epoch in range(20):
    running_loss = 0.0
    mlp.train()  # Set model to training mode
    for data, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass through MLP to get logits
        outputs = mlp(data, training=True)
        
        # Calculate loss using CrossEntropyLoss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        running_loss += loss.item()

    # Calculate accuracy after each epoch
    train_acc = compute_accuracy(train_loader)
    test_acc = compute_accuracy(test_loader)
    print(f"Epoch [{epoch+1}] | Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc L2 {train_acc:.4f} | Test Acc L2 {test_acc:.4f}")
# optimizer = torch.optim.Adam(P.parameters(), lr=0.001)
# for epoch in range(20):
#     running_loss = 0.0
#     for data, labels in train_loader:
#         optimizer.zero_grad()
#         x = ltn.Variable("x", data)
#         x_benign = ltn.Variable("x_benign", data[labels == 5])
#         l_x = ltn.Constant(torch.nn.functional.one_hot(labels, num_classes=6))
#         l_benign = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1]))
#         if x_benign.value.size(0) != 0:
#             sat_agg = SatAgg(Forall(x_benign, P(x_benign, l_benign)))
#         else:
#             sat_agg = torch.tensor(0.0, requires_grad=True, device=device)  # Set as a tensor on the correct device

#         loss = 1. - sat_agg
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     train_acc = compute_accuracy(train_loader)
#     test_acc = compute_accuracy(test_loader)
#     print(f"Epoch [{epoch+1}] | Loss: {running_loss/len(train_loader):.4f} | "
#           f"Train Acc L2 {train_acc:.4f} | Test Acc L2 {test_acc:.4f}")



# print("Making predictions...")
# y_test = test_data['label_L2']
# # y_test = test_data['label']
# y_pred = model.predict(test_data[X_columns])


# print("Evaluating model...")
# # 计算和打印评估指标
# print('Accuracy Score:', accuracy_score(y_test, y_pred))
# print('Recall Score (Macro):', recall_score(y_test, y_pred, average='macro'))
# print('Precision Score (Macro):', precision_score(y_test, y_pred, average='macro'))
# print('F1 Score (Macro):', f1_score(y_test, y_pred, average='macro'))
# # print('Recall Scores by Class:', recall_score(y_test, y_pred, average=None))
# # print('Precision Scores by Class:', precision_score(y_test, y_pred, average=None))
# # print('F1 Scores by Class:', f1_score(y_test, y_pred, average=None))
# print("Model evaluation complete.")
