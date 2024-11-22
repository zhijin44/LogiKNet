import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Example Dataset
# from sklearn.datasets import load_iris
# data = load_iris()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = data.target

# 特征列名称
X_columns = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance',
    'Variance', 'Weight'
]

#########################################Compute SHAP##########################################
# # 定义处理后的数据文件路径
# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_s_4_11.csv'
# processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_tiny_4_11.csv'
# # processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_small_train.csv'
# # processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_small_test.csv'


# # 数据预处理
# print("Loading processed data...")
# training_data = pd.read_csv(processed_train_file)
# test_data = pd.read_csv(processed_test_file)


# label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 0, "MQTT-DDoS-Publish_Flood": 1, 
#                     "MQTT-DoS-Connect_Flood": 2, "MQTT-DoS-Publish_Flood": 3,
#                     "MQTT-Malformed_Data": 4, "benign": 5, 
#                     "Recon-OS_Scan": 6, "Recon-Port_Scan": 7,
#                     "Recon-VulScan": 8, "Recon-Ping_Sweep": 9,
#                     "arp_spoofing": 10}
# train_label_L2 = training_data.pop("label_L2").map(label_L2_mapping)
# test_label_L2 = test_data.pop("label_L2").map(label_L2_mapping)


# print("Scaling data...")
# scaler = StandardScaler()
# train_data_scaled = scaler.fit_transform(training_data[X_columns])
# test_data_scaled = scaler.transform(test_data[X_columns])

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_data_scaled, test_data_scaled, train_label_L2, test_label_L2

# # Train a Model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Initialize SHAP Explainer
# explainer = shap.TreeExplainer(model)  # Explicitly use TreeExplainer for RandomForest

# # Compute SHAP values
# shap_values = explainer.shap_values(X_test, check_additivity=False)  # List of SHAP values per class


# # Save SHAP values and corresponding data
# output_file = "shap_values.pkl"
# with open(output_file, "wb") as f:
#     pickle.dump({
#         "shap_values": shap_values,  # SHAP values (list of arrays)
#         "X_test": X_test,            # Test data (to ensure features are available for later plotting)
#         "class_names": model.classes_  # Class names, if available
#     }, f)

# print(f"SHAP values saved to '{output_file}'")


# print("Saving SHAP summary plot for class 1...")
# plt.figure()
# shap.summary_plot(shap_values[1], X_test, show=False)
# plt.savefig("shap_summary_class_1.png", bbox_inches='tight')
# plt.close()


# print("Saving SHAP mean summary plot across all classes...")
# shap_values_mean = np.mean(np.abs(shap_values), axis=0) 
# plt.figure()
# shap.summary_plot(shap_values_mean, X_test, show=False)
# plt.savefig("shap_mean_summary_all_classes.png", bbox_inches='tight')
# plt.close()


# print("Saving SHAP bar plot for class 1...")
# plt.figure()
# shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
# plt.savefig("shap_bar_class_1.png", bbox_inches='tight')
# plt.close()

#########################################Eva SHAP##########################################
# Load SHAP values and test data
with open("shap_values.pkl", "rb") as f:
    data = pickle.load(f)

shap_values = data["shap_values"]
X_test = data["X_test"]
class_names = data.get("class_names", None)
print("SHAP values successfully loaded.")


# Compute mean SHAP values for each class and print the top 5 features
print("Top 5 mean SHAP values for each class:")

for class_index, class_shap_values in enumerate(shap_values):  # Loop over all classes
    mean_shap = np.abs(class_shap_values).mean(axis=0)  # Mean absolute SHAP values for each feature
    
    # Sort features by mean SHAP values in descending order
    sorted_indices = np.argsort(mean_shap)[::-1]  # Get indices of top features
    top_features = [(X_columns[i], mean_shap[i]) for i in sorted_indices[:5]]  # Top 5 features
    
    # Print results for the class
    print(f"\nClass {class_index}:")
    for feature, value in top_features:
        print(f"  {feature}: {value:.4f}")





