import ltn.fuzzy_ops
import torch
import pandas as pd
import ltn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import MLP, LogitsToPredicate, DataLoader
import custom_fuzzy_ops as custom_fuzzy_ops
import logging
import sys
import os

# Set up logging
log_file = "/home/zyang44/Github/baseline_cicIOT/LTNtorch/LTN_IoV/training_log.txt"
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
    for data, labels in loader:
        x = ltn.Variable("x", data)
        x_syn_flood = ltn.Variable("x_syn_flood", data[labels == 0])
        x_tcp_flood = ltn.Variable("x_tcp_flood", data[labels == 1])
        x_none = ltn.Variable("x_none", data[labels == 2])
        x_cryptojacking = ltn.Variable("x_cryptojacking", data[labels == 3])
        x_syn_stealth = ltn.Variable("x_syn_stealth", data[labels == 4])
        x_vuln_scan = ltn.Variable("x_vuln_scan", data[labels == 5]) 
        x_Backdoor = ltn.Variable("x_Backdoor", data[labels == 6])

        x_cryptojacking_current = ltn.Variable("x_cryptojacking_current", data[labels == 3])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_syn_flood, l_syn_flood),
            (x_tcp_flood, l_tcp_flood),
            (x_none, l_none),
            (x_cryptojacking, l_cryptojacking),
            (x_syn_stealth, l_syn_stealth),
            (x_vuln_scan, l_vuln_scan),
            (x_Backdoor, l_Backdoor)
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label)))

        mean_sat += SatAgg(*valid_forall_expressions)
    # In the loop: mean_sat accumulates the satisfaction levels for all the logical rules across the batches.
    # After the loop: mean_sat becomes the average satisfaction level of the logical rules over the entire dataset.
    mean_sat /= len(loader)
    return mean_sat

def compute_metrics_hierarchy(loader, model, class_names_L1, class_names_L2):
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
    mean_accuracy = 0.0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        predictions = mlp(data).detach()
        # Stay on GPU, use torch operations
        predictions = torch.argmax(predictions, dim=1)
        mean_accuracy += (predictions == labels).float().mean().item()

    return mean_accuracy / len(loader)

def plot_confusion_matrix(loader, model, class_names, filename="LTN_IoV_confMat.png"):
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

#####################Preprocess#################################
file_path = '/home/zyang44/Github/baseline_cicIOT/IoV_power_L.csv'
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

print("Data processing and scaling done.")

#####################Setting#################################
# define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
# And = ltn.Connective(custom_fuzzy_ops.AndProd())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
# higher is p and easier is the existential quantification to be satisfied, 
# while harder is the universal quantification to be satisfied.
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
SatAgg = ltn.fuzzy_ops.SatAgg()

labels = ['syn_flood', 'tcp_flood', 'none', 'cryptojacking', 'syn_stealth', 'vuln_scan', 'Backdoor']
constants = [ltn.Constant(torch.tensor([1 if i == j else 0 for i in range(len(labels))])) for j in range(len(labels))]
l_syn_flood, l_tcp_flood, l_none, l_cryptojacking, l_syn_stealth, l_vuln_scan, l_Backdoor = constants

print("LTN setting done.")

#####################Training#################################
mlp = MLP(layer_sizes=(5, 32, 64, 7)).to(device)
P = ltn.Predicate(LogitsToPredicate(mlp))
# P_1 = ltn.Predicate(LogitsToPredicate(mlp))

batch_size = 64
train_loader = DataLoader(train_data, train_label, batch_size, shuffle=True)
test_loader = DataLoader(test_data, test_label, batch_size, shuffle=False)

print("Start training...")
optimizer = torch.optim.Adam(P.parameters(), lr=0.001)
# # Single optimizer for all parameters
# optimizer = torch.optim.Adam([
#     {'params': P.parameters()},
#     {'params': P_1.parameters()}
# ], lr=0.001)

for epoch in range(1):
    train_loss = 0.0

    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        x = ltn.Variable("x", data)
        x_syn_flood = ltn.Variable("x_syn_flood", data[labels == 0])
        x_tcp_flood = ltn.Variable("x_tcp_flood", data[labels == 1])
        x_none = ltn.Variable("x_none", data[labels == 2])
        x_cryptojacking = ltn.Variable("x_cryptojacking", data[labels == 3])
        x_syn_stealth = ltn.Variable("x_syn_stealth", data[labels == 4])
        x_vuln_scan = ltn.Variable("x_vuln_scan", data[labels == 5]) 
        x_Backdoor = ltn.Variable("x_Backdoor", data[labels == 6])

        # rules - single class exclusive
        valid_forall_expressions = []
        variables_labels = [
            (x_syn_flood, l_syn_flood),
            (x_tcp_flood, l_tcp_flood),
            (x_none, l_none),
            (x_cryptojacking, l_cryptojacking),
            (x_syn_stealth, l_syn_stealth),
            (x_vuln_scan, l_vuln_scan),
            (x_Backdoor, l_Backdoor)
        ]
        for variable, label in variables_labels:
            if variable.value.size(0) != 0:
                valid_forall_expressions.append(Forall(variable, P(variable, label)))
        
        
        sat_agg = SatAgg(*valid_forall_expressions)
        loss = 1. - sat_agg     # loss = -torch.mean(sat_agg)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    train_sat, test_sat = compute_sat_level(train_loader), compute_sat_level(test_loader)
    train_acc, test_acc = compute_accuracy(train_loader), compute_accuracy(test_loader)
    print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train SAT: {train_sat:.4f}, Test SAT: {test_sat:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    logging.info(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train SAT: {train_sat:.4f}, Test SAT: {test_sat:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

#####################Evaluation#################################
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
    return report

class_names = list(attack_mapping.keys())
report = print_metrics(test_loader, mlp, class_names)
logging.info(f"\n {report}")