####################LIME#################################
from lime.lime_tabular import LimeTabularExplainer

# 特征列名称
X_columns = [
    'Header_Length', # 3 
    'Protocol Type', 'Duration', 
    'Rate', 
    'Srate', 'Drate',
    # 'fin_flag_number', 
    # 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    # 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'syn_count',
    'ack_count', # 5
    'fin_count', # 2
    # 'rst_count', 
    # 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    # 'SMTP', 'SSH', 'IRC', 
    # 'TCP', 
    # 'UDP', 'DHCP', 
    # 'ARP', 
    # 'ICMP', 'IGMP', 'IPv',
    # 'LLC', 'Tot sum', 'Min', 'Max', 
    # 'AVG', 
    # 'Std', 'Tot size', 
    'IAT', # 1
    # 'Number', 'Magnitue', 'Radius', 'Covariance', 
    # 'Variance', 
    # 'Weight' # 4
]


class_names_L1 = ["MQTT", "Benign", "Recon", "ARP_Spoofing"]
class_names_L2 = [
    "MQTT-DDoS-Connect_Flood", 
    "MQTT-DDoS-Publish_Flood", 
    "MQTT-DoS-Connect_Flood", 
    "MQTT-DoS-Publish_Flood", 
    "MQTT-Malformed_Data",
    "benign",
    'Recon-Port_Scan',
    'Recon-OS_Scan',
    'arp_spoofing'
]



# Prepare training data and feature names
X_train = train_data.cpu().numpy()
class_names_all = class_names_L1 + class_names_L2  # Replace with actual feature names

# Create LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=X_columns,
    class_names=class_names_all,  # Class names for L2 predictions
    mode="classification"
)

# Select a single instance for explanation
test_instance = test_data[0].cpu().numpy()  # Example instance
true_label_L2 = test_label_L2[0].item()  # True label for L2

# Define prediction function
def predict_fn(data):
    data_tensor = torch.tensor(data).float().to(device)  # Convert to tensor and move to device
    logits = mlp(data_tensor).detach().cpu().numpy()  # Get logits from MLP
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
    return probabilities

# Generate explanation for the instance
explanation = explainer.explain_instance(
    data_row=test_instance,
    predict_fn=predict_fn
)

# Visualize the explanation
explanation.save_to_file("lime_explanation.html")  # Save as HTML for external viewing