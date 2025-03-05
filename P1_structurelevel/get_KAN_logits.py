from kan import KAN
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

X_columns = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 
    # 'Drate',
    # 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    # 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
    # 'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    # 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 
    'IPv','LLC', 
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance',
    'Variance', 'Weight'
]

Y_columns = ['label_L1']

label_L1_mapping = {"MQTT": 0, "Benign": 1, "Recon": 2, "ARP_Spoofing": 3}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 4, "MQTT-DDoS-Publish_Flood": 5, 
                    "MQTT-DoS-Connect_Flood": 6, "MQTT-DoS-Publish_Flood": 7,
                    "MQTT-Malformed_Data": 8, "benign": 9, 
                    "Recon-OS_Scan": 10, "Recon-Port_Scan": 11,
                    "arp_spoofing": 12}


# Read the CSV file
df = pd.read_csv('/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_s_4_11.csv')

df['label_L1'] = df['label_L1'].map(label_L1_mapping)
df['label_L2'] = df['label_L2'].map(label_L1_mapping)

# 90% as training set and 10% left as test set
train_size = int(len(df) * 0.9)
train_df, test_df = df.iloc[:train_size, :], df.iloc[train_size:, :]

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_df[X_columns])
test_X_scaled = scaler.transform(test_df[X_columns])
train_y = train_df[Y_columns].values.ravel()
test_y = test_df[Y_columns].values.ravel()

# take Y_columns as the label, and transfering to one-hot coded
dataset = {
    'train_input': torch.tensor(train_X_scaled, dtype=torch.float32, device=device),
    'train_label': F.one_hot(torch.tensor(train_y, dtype=torch.long, device=device), num_classes=4),
    'test_input': torch.tensor(test_X_scaled, dtype=torch.float32, device=device),
    'test_label': F.one_hot(torch.tensor(test_y, dtype=torch.long, device=device), num_classes=4)
}
print("Data prepared.",
      f"Train set: {dataset['train_input'].shape, dataset['train_label'].shape}",
      f"Test set: {dataset['test_input'].shape, dataset['test_label'].shape}", sep="\n")


# create a KAN
kan = KAN(width=[20, 10, 4], grid=5, k=3, seed=42, device=device)

results = kan.fit(dataset, opt="Adam", steps=3)

train_logits = kan(dataset['train_input'])
test_logits = kan(dataset['test_input'])

import ltn
import ltn.fuzzy_ops
from utils import MLP, LogitsToPredicate, DataLoaderMulti

l_MQTT = ltn.Constant(torch.tensor([1, 0, 0, 0, ]))
l_Benign = ltn.Constant(torch.tensor([0, 1, 0, 0]))
l_Recon = ltn.Constant(torch.tensor([0, 0, 1, 0]))
l_ARP_Spoofing = ltn.Constant(torch.tensor([0, 0, 0, 1]))



