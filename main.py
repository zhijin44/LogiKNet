import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
from models import GINModel

BATCH_SIZE = 64

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file_path = 'CIC_IOT/part-00004.csv'
file_path = 'CIC_IOT/part-00000.csv'
df = pd.read_csv(file_path)
X = df.drop('label', axis=1)
y = df[['label']].values

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=seed)

# Now you have:
# X_train, y_train: Features and labels for the training set
# X_test, y_test: Features and labels for the testing set
# Optionally, check the shapes to confirm the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Create DataLoader instances
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# class MLP(nn.Module):
#     """Model that returns logits."""
#
#     def __init__(self, n_classes=34, hidden_layer_sizes=(32, 64, 32)):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#
#         # Input layer size
#         input_size = 46  # Adjusted for 46 input features
#
#         # Create the first hidden layer separately to handle the input feature size
#         if hidden_layer_sizes:
#             self.layers.append(nn.Linear(in_features=input_size, out_features=hidden_layer_sizes[0]))
#             self.layers.append(nn.ELU())
#             self.layers.append(nn.Dropout(p=0.2))
#
#         # Create additional hidden layers
#         for i in range(1, len(hidden_layer_sizes)):
#             self.layers.append(nn.Linear(in_features=hidden_layer_sizes[i - 1], out_features=hidden_layer_sizes[i]))
#             self.layers.append(nn.ELU())
#             self.layers.append(nn.Dropout(p=0.2))
#
#         # Add the final classification layer
#         self.classifier = nn.Linear(in_features=hidden_layer_sizes[-1], out_features=n_classes)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         x = self.classifier(x)
#         return x


model = GINModel(num_node_features=46, num_classes=34, dim_h=128, layers=3)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training the model
model = model.to(device)
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)   # move data to device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluating the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # move data to device
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

print(f'Accuracy of the network on the test data: {100 * correct / total}%')
