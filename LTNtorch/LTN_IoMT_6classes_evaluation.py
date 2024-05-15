import torch
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
from LTN_IoMT_6classes import MLP, DataLoader


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


# 创建模型实例并加载权重
mlp = MLP().to(device)
mlp.load_state_dict(torch.load('LTN_reduce.pth'))
mlp.eval()

# 加载test数据和test loader
processed_test_file = '../CIC_IoMT/6classes/6classes_1700_test.csv'
test_data = pd.read_csv(processed_test_file)
test_labels = test_data.pop("label")

label_mapping = {"ARP_Spoofing": 0, "Benign": 1, "MQTT": 2, "Recon": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5}
test_labels = test_labels.map(label_mapping)

scaler = StandardScaler()
test_data_scaled = scaler.transform(test_data)
test_data_scaled = torch.tensor(test_data_scaled).float()
test_labels = torch.tensor(test_labels.to_numpy()).long()

test_data = test_data_scaled.to(device)
test_labels = test_labels.to(device)
batch_size = 512
test_loader = DataLoader(test_data, test_labels, batch_size, shuffle=False)

# 收集预测和标签，计算评估指标
all_labels, all_probabilities = collect_predictions_and_labels(test_loader, mlp)

# Binarize labels for each class
n_classes = 6  # Update this with your number of classes
labels_binarized = label_binarize(all_labels.numpy(), classes=[*range(n_classes)])

# Compute Precision-Recall for each class
precision = dict()
recall = dict()
thresholds = dict()
for i in range(n_classes):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(labels_binarized[:, i],
                                                                    all_probabilities[:, i])

# Define class names if you have them
class_names = ["ARP_Spoofing", "Benign", "MQTT", "Recon", "TCP_IP-DDOS", "TCP_IP-DOS"]

# Plot PR curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label='PR curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], auc(recall[i], precision[i])))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve per class')
plt.legend(loc="best")
plt.show()
