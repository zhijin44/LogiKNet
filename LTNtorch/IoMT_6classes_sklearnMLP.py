import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 假设这里是你的新数据集路径
# processed_train_file = '../CIC_IoMT/6classes/processed_train_data_6classes.csv'
# processed_test_file = '../CIC_IoMT/6classes/processed_test_data_6classes.csv'
processed_train_file = '../CIC_IoMT/6classes/6classes_15k_train.csv'
processed_test_file = '../CIC_IoMT/6classes/6classes_1700_test.csv'

# 加载数据集
train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 假设数据集中的标签列名为"label"
train_labels = train_data.pop("label")
test_labels = test_data.pop("label")
# print(train_labels.head())
# print(test_labels.head())

# 将标签映射到整数，适用于6个类别的场景
label_mapping = {"ARP_Spoofing": 0, "Benign": 1, "MQTT": 2, "Recon": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5}
train_labels = train_labels.map(label_mapping)
test_labels = test_labels.map(label_mapping)

# 使用 StandardScaler 对数据进行缩放
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 将缩放后的数据和标签转换为Tensor
train_data_scaled = torch.tensor(train_data_scaled).float()
test_data_scaled = torch.tensor(test_data_scaled).float()
train_labels = torch.tensor(train_labels.to_numpy()).long()
test_labels = torch.tensor(test_labels.to_numpy()).long()

# 定义设备并移动数据和标签到设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data, test_data = train_data_scaled.to(device), test_data_scaled.to(device)
# train_data, test_data = train_data.to(device), test_data.to(device)
train_labels, test_labels = train_labels.to(device), test_labels.to(device)

# 定义常量并移动到设备，适用于6个类别
l_A = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0]).to(device))
l_B = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0]).to(device))
l_C = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0]).to(device))
l_D = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0]).to(device))
l_E = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0]).to(device))
l_F = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1]).to(device))


# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes=(45, 32, 32, 6)):
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


# 创建模型实例并移动到设备
# mlp = MLP().to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
#####################################################################
mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=200, activation='relu', solver='adam', random_state=1,
                          verbose=True).to(device)
#####################################################################
P = ltn.Predicate(LogitsToPredicate(mlp))

# we define the connectives, quantifiers, and the SatAgg
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()


# define utility classes and functions

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


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, labels in loader:
        x_A = ltn.Variable("x_A", data[labels == 0])
        x_B = ltn.Variable("x_B", data[labels == 1])
        x_C = ltn.Variable("x_C", data[labels == 2])
        x_D = ltn.Variable("x_D", data[labels == 3])
        x_E = ltn.Variable("x_E", data[labels == 4])
        x_F = ltn.Variable("x_F", data[labels == 5])
        ###########################################################
        valid_forall_expressions = []
        variables_labels = [(x_A, l_A),
                            (x_B, l_B),
                            (x_C, l_C),
                            (x_D, l_D),
                            (x_E, l_E),
                            (x_F, l_F),
                            ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
        mean_sat += SatAgg(*valid_forall_expressions)
        ##############################################################
    mean_sat /= len(loader)
    return mean_sat


# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    # mean_accuracy = 0.0
    # for data, labels in loader:
    #     # 确保数据在正确的设备上
    #     data, labels = data.to(device), labels.to(device)
    #     predictions = mlp(data).detach().numpy()
    #     predictions = np.argmax(predictions, axis=1)
    #     mean_accuracy += accuracy_score(labels, predictions)
    #
    # return mean_accuracy / len(loader)

    mean_accuracy = 0.0
    for data, labels in loader:
        # 确保数据在正确的设备上
        data, labels = data.to(device), labels.to(device)
        predictions = mlp(data).detach()
        predictions = predictions.cpu().numpy()  # 先移动到CPU，然后转换为NumPy数组
        predictions = np.argmax(predictions, axis=1)
        mean_accuracy += accuracy_score(labels.cpu().numpy(), predictions)  # 同样，确保labels也在CPU上

    return mean_accuracy / len(loader)


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


# create train and test loader
batch_size = 512
train_loader = DataLoader(train_data, train_labels, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_labels, batch_size, shuffle=False)

# Learning
optimizer = torch.optim.Adam(P.parameters(), lr=0.0001)

for epoch in range(50):
    train_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x_A = ltn.Variable("x_A", data[labels == 0])  # class A examples
        x_B = ltn.Variable("x_B", data[labels == 1])  # class B examples
        x_C = ltn.Variable("x_C", data[labels == 2])  # class C examples
        x_D = ltn.Variable("x_D", data[labels == 3])
        x_E = ltn.Variable("x_E", data[labels == 4])
        x_F = ltn.Variable("x_F", data[labels == 5])

        ######################################################
        # List to hold valid Forall expressions
        valid_forall_expressions = []
        variables_labels = [(x_A, l_A),
                            (x_B, l_B),
                            (x_C, l_C),
                            (x_D, l_D),
                            (x_E, l_E),
                            (x_F, l_F),
                            ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label, training=True)))
            # else:
            #     print(f"{variable.free_vars[0]} is empty, no this class in batch {batch_idx}")
        #########################################################
        sat_agg = SatAgg(*valid_forall_expressions)
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 20 epochs of training
    if epoch % 1 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f"
              % (epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
                 compute_accuracy(train_loader), compute_accuracy(test_loader)))
        ###############################################################################################
        precision, recall, f1 = compute_metrics(test_loader, mlp)
        print(f"Macro Recall: {recall.mean():.4f}, Macro Precision: {precision.mean():.4f}, Macro F1-Score: {f1.mean():.4f}")


class_names = ["ARP_Spoofing", "Benign", "MQTT", "Recon", "TCP_IP-DDOS", "TCP_IP-DOS"]
precision, recall, f1 = compute_metrics(test_loader, mlp)
print("Scores by Class:")
for i, class_name in enumerate(class_names):
    print(f"Class {class_name}: Recall: {recall[i]:.6f}, Precision: {precision[i]:.6f}, F1: {f1[i]:.6f}")


###############################################################################################

# 训练循环结束后保存模型
# model_save_path = 'LTN_reduce.pth'
# torch.save(mlp.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")

def plot_pr_curves(labels, probabilities, class_names, save_path):
    # Binarize labels for each class
    labels_binarized = label_binarize(labels, classes=list(range(len(class_names))))

    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(labels_binarized[:, i], probabilities[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'Class {class_name} (AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve per class')
    plt.legend(loc="best")
    plt.grid(True)
    # Save the plot to a file
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free up memory


def collect_predictions_and_labels(loader, model):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            probabilities = torch.softmax(model(data), dim=1)  # Assuming model outputs raw logits
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

    # Concatenate all batches
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)
    return all_labels, all_probabilities


# 在训练循环结束后绘制PR曲线
all_labels, all_probabilities = collect_predictions_and_labels(test_loader, mlp)
save_path = "outputs/LTN_6classes_reduce_PR_curve.png"  # 设定保存路径和文件名
# save_path = "outputs/LTN_6classes_PR_curve.png"
plot_pr_curves(all_labels.numpy(), all_probabilities.numpy(), class_names, save_path)
