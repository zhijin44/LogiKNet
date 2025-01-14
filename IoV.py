import pandas as pd
import numpy as np
import os
import torch
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


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

from sklearn.model_selection import train_test_split

# Load the balanced dataset
file_path = '/home/zyang44/Github/baseline_cicIOT/IoV_power_L.csv'  # Replace with your actual file path
balanced_data = pd.read_csv(file_path)

attack_mapping = {'syn-flood': 0, 'tcp-flood': 1, 'none': 2, 'cryptojacking': 3, 'syn-stealth': 4, 'vuln-scan': 5, 'Backdoor': 6}
state_mapping = {'idle': 0, 'charging': 1}
# Map the State and Attack columns
balanced_data['State'] = balanced_data['State'].map(state_mapping)
balanced_data['Attack'] = balanced_data['Attack'].map(attack_mapping)

# numerical_ranges = balanced_data.describe().loc[['min', 'max']]

# Split the data into train (70%) and test (30%) sets
train_data, test_data = train_test_split(balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['Attack'])
# Extract the Attack column as the label
train_label, test_label = train_data.pop('Attack'), test_data.pop('Attack')

print("Scaling data...")
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 定义设备并移动数据和标签到设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = torch.tensor(train_data_scaled).float().to(device)
test_data = torch.tensor(test_data_scaled).float().to(device)
train_label = torch.tensor(train_label.to_numpy()).long().to(device)
test_label = torch.tensor(test_label.to_numpy()).long().to(device)

# 创建模型实例并移动到设备 
mlp = MLP(layer_sizes=(5, 32, 32, 7)).to(device)  # 输出的数值可以被理解为模型对每个类别的信心水平
# create train and test loader (train_sex_labels, train_color_labels)
batch_size = 64
train_loader = DataLoader(train_data, train_label, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label, batch_size, shuffle=False)

print("Training model...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
for epoch in range(35):
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
          f"Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")

################################################################################
# 1. Function to print precision, recall, and F1 scores
def print_metrics(loader, model, class_names):
    all_preds = []
    all_labels = []
    
    # Collect predictions and true labels
    for data, labels in loader:
        outputs = model(data).detach().cpu().numpy()
        preds = np.argmax(outputs, axis=-1)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,  # Handle any undefined metrics
    )
    print("Classification Report:\n")
    print(report)

class_names = list(attack_mapping.keys())
print_metrics(test_loader, mlp, class_names)