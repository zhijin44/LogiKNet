import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

X_columns = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

# Load the saved model and scaler
# model_path = 'CIC_IoMT/6classes/mlp_classifier_model_6classes.joblib'
# scaler_path = 'CIC_IoMT/6classes/scaler_6classes.joblib'
model_path = 'CIC_IoMT/6classes/mlp_classifier_model_6classes_reduced.joblib'
scaler_path = 'CIC_IoMT/6classes/scaler_6classes_reduced.joblib'
model = load(model_path)
scaler = load(scaler_path)
print("Model and scaler loaded.")

# Load the processed test data
# test_data_path = 'CIC_IoMT/6classes/processed_test_data_6classes.csv'
test_data_path = 'CIC_IoMT/6classes/6classes_1700_test.csv'
test_data = pd.read_csv(test_data_path)
X_test = test_data[X_columns]
y_test = test_data['label']

# dynamically creating a class_labels_dict from y_test
unique_labels = np.unique(y_test)
class_labels_dict = {label: label for label in unique_labels}

# Apply the scaler to the test data
X_test_scaled = scaler.transform(X_test)
print("Test data scaled.")

# Make predictions using the scaled test data
y_pred = model.predict(X_test_scaled)
print("Predictions made.")

# Evaluate the model
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Recall Score (Macro):', recall_score(y_test, y_pred, average='macro'))
print('Precision Score (Macro):', precision_score(y_test, y_pred, average='macro'))
print('F1 Score (Macro):', f1_score(y_test, y_pred, average='macro'))

# Compute and display the confusion matrix
# Create the output directory if it doesn't exist
output_dir = './outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_labels, output_file):
    plt.figure(figsize=(10, 7))  # Adjust size as needed
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # Ensure the directory exists
    output_path = os.path.join(output_dir, output_file)
    # Save the plot
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')  # Save as high-resolution PNG file
    plt.show()


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
# Plot the confusion matrix
plot_confusion_matrix(cm, np.unique(y_test), output_file='reduce_confusion_matrix_6classes.png')

# Compute scores for each class
recall_scores = recall_score(y_test, y_pred, average=None, labels=np.unique(y_test), zero_division=0)
precision_scores = precision_score(y_test, y_pred, average=None, labels=np.unique(y_test), zero_division=0)
f1_scores = f1_score(y_test, y_pred, average=None, labels=np.unique(y_test))

# Get unique labels from y_test for display
unique_labels = np.unique(y_test)

# Display the scores for each class along with the label name
print("Scores by Class:")
for i, label in enumerate(unique_labels):
    print(f"Class {label}: Recall: {recall_scores[i]}, Precision: {precision_scores[i]}, F1: {f1_scores[i]}")

print("Model evaluation complete.")

from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

# Assuming y_test, X_test_scaled, model, and unique_labels are already defined

# Binarize the output labels for multi-class precision-recall curve calculation
y_test_bin = label_binarize(y_test, classes=unique_labels)
n_classes = y_test_bin.shape[1]

# Predict probabilities for each class
y_score = model.predict_proba(X_test_scaled)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# Compute micro-average Precision-Recall curve and area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(),
                                                                y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_bin, y_score,
                                                     average="micro")

# Plotting Precision-Recall curves for each class
plt.figure(figsize=(12, 8))
lw = 2  # Line width
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                   ''.format(class_labels_dict[unique_labels[i]], average_precision[i]))

plt.plot([0, 1], [0.1, 0.1], 'k--', lw=lw)  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall')
plt.legend(loc="lower left")
plt.show()

