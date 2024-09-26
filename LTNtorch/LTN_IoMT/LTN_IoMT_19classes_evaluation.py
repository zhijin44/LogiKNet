import torch
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import joblib
from utils import MLP, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    plt.close()  # Close the figure to free up memory


def plot_confusion_matrix(labels, probabilities, class_names, save_path):
    # Get the predicted labels by taking the class with the highest probability
    predicted_labels = np.argmax(probabilities, axis=1)
    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(labels, predicted_labels)
    
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot to a file
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


# 创建模型实例并加载权重
mlp = MLP(layer_sizes=(46, 64, 32, 19)).to(device)
mlp.load_state_dict(torch.load('/home/zyang44/Github/baseline_cicIOT/LTNtorch/LTN_IoMT/LTN_reduce_19classes.pth'))
mlp.eval()

# 加载test数据和test loader
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_test_data.csv'
test_data = pd.read_csv(processed_test_file)
test_labels = test_data.pop("label_L2")

label_L1_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 0, "MQTT-DDoS-Publish_Flood": 1, "MQTT-DoS-Connect_Flood": 2, "MQTT-DoS-Publish_Flood": 3, "MQTT-Malformed_Data": 4,
                 "Recon-Port_Scan": 5, "Recon-OS_Scan": 6, "Recon-VulScan": 7, "Recon-Ping_Sweep": 8,
                 "TCP_IP-DDoS-TCP": 9, "TCP_IP-DDoS-ICMP": 10,  "TCP_IP-DDoS-SYN": 11, "TCP_IP-DDoS-UDP": 12,
                 "TCP_IP-DoS-TCP": 13, "TCP_IP-DoS-ICMP": 14, "TCP_IP-DoS-SYN": 15, "TCP_IP-DoS-UDP": 16,
                 "benign": 17, "arp_spoofing": 18}
test_data["label_L1"] = test_data["label_L1"].map(label_L1_mapping)
test_labels = test_labels.map(label_L2_mapping)

scaler = joblib.load('/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/scaler_reduced_46features.joblib')
test_data_scaled = scaler.transform(test_data)
test_data_scaled = torch.tensor(test_data_scaled).float()
test_labels = torch.tensor(test_labels.to_numpy()).long()

test_data = test_data_scaled.to(device)
test_labels = test_labels.to(device)
batch_size = 64
test_loader = DataLoader(test_data, test_labels, batch_size, shuffle=False)

# 收集预测和标签，计算评估指标
all_labels, all_probabilities = collect_predictions_and_labels(test_loader, mlp)
class_names = [
    "MQTT-DDoS-Connect_Flood", 
    "MQTT-DDoS-Publish_Flood", 
    "MQTT-DoS-Connect_Flood", 
    "MQTT-DoS-Publish_Flood", 
    "MQTT-Malformed_Data",
    "Recon-Port_Scan", 
    "Recon-OS_Scan", 
    "Recon-VulScan", 
    "Recon-Ping_Sweep", 
    "TCP_IP-DDoS-TCP", 
    "TCP_IP-DDoS-ICMP",  
    "TCP_IP-DDoS-SYN", 
    "TCP_IP-DDoS-UDP", 
    "TCP_IP-DoS-TCP", 
    "TCP_IP-DoS-ICMP", 
    "TCP_IP-DoS-SYN", 
    "TCP_IP-DoS-UDP", 
    "benign", 
    "arp_spoofing"
]

# draw PR curve
save_path = '/home/zyang44/Github/baseline_cicIOT/LTNtorch/outputs/temp_pr.png'
plot_pr_curves(all_labels.numpy(), all_probabilities.numpy(), class_names, save_path)

# draw confusion matrix
save_path = '/home/zyang44/Github/baseline_cicIOT/LTNtorch/outputs/temp_conf.png'
plot_confusion_matrix(all_labels.numpy(), all_probabilities.numpy(), class_names, save_path)