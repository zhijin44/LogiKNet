import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings('ignore')

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
            yield data, labels


def compute_accuracy(loader):
    mean_accuracy_L2 = 0.0
    for data, label_L2 in loader:
        predictions = mlp(data).detach().cpu().numpy()
        pred_L2 = np.argmax(predictions, axis=-1)
        true_L2 = label_L2.cpu().numpy()
        accuracy_L2 = np.mean(pred_L2 == true_L2)
        mean_accuracy_L2 += accuracy_L2
    mean_accuracy_L2 /= len(loader)
    return mean_accuracy_L2

from sklearn.model_selection import train_test_split

# Load the balanced dataset
file_path = 'balanced_dataset_l.csv'
balanced_data = pd.read_csv(file_path)

attack_mapping = {'syn-flood': 0, 'tcp-flood': 1, 'none': 2, 'cryptojacking': 3, 'syn-stealth': 4, 'vuln-scan': 5, 'Backdoor': 6}
state_mapping = {'idle': 0, 'charging': 1}
balanced_data['State'] = balanced_data['State'].map(state_mapping)
balanced_data['Attack'] = balanced_data['Attack'].map(attack_mapping)

# Split the data into train (70%) and test (30%) sets
train_data, test_data = train_test_split(balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['Attack'])
train_label, test_label = train_data.pop('Attack'), test_data.pop('Attack')

print("Scaling data...")
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = torch.tensor(train_data_scaled).float().to(device)
test_data = torch.tensor(test_data_scaled).float().to(device)
train_label = torch.tensor(train_label.to_numpy()).long().to(device)
test_label = torch.tensor(test_label.to_numpy()).long().to(device)

mlp = MLP(layer_sizes=(5, 32, 64, 7)).to(device)
batch_size = 64
train_loader = DataLoader(train_data, train_label, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label, batch_size, shuffle=False)

print("Training model...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
for epoch in range(30):
    running_loss = 0.0
    mlp.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = mlp(data, training=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = compute_accuracy(train_loader)
    test_acc = compute_accuracy(test_loader)
    print(f"Epoch [{epoch+1}] | Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")

################################################################################
