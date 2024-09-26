import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# 加载数据集
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_train_data.csv'

train_data = pd.read_csv(processed_train_file)

# 将标签映射到整数，适用于19个类别的场景 ('label_L1' remains in the train_data and test_data)
label_L1_mapping = {"Benign": 0, "MQTT": 1, "Recon": 2, "ARP_Spoofing": 3, "TCP_IP-DDOS": 4, "TCP_IP-DOS": 5}
label_L2_mapping = {"MQTT-DDoS-Connect_Flood": 0, "MQTT-DDoS-Publish_Flood": 1, "MQTT-DoS-Connect_Flood": 2, "MQTT-DoS-Publish_Flood": 3, "MQTT-Malformed_Data": 4,
                 "Recon-Port_Scan": 5, "Recon-OS_Scan": 6, "Recon-VulScan": 7, "Recon-Ping_Sweep": 8,
                 "TCP_IP-DDoS-TCP": 9, "TCP_IP-DDoS-ICMP": 10,  "TCP_IP-DDoS-SYN": 11, "TCP_IP-DDoS-UDP": 12,
                 "TCP_IP-DoS-TCP": 13, "TCP_IP-DoS-ICMP": 14, "TCP_IP-DoS-SYN": 15, "TCP_IP-DoS-UDP": 16,
                 "benign": 17, "arp_spoofing": 18}
# 应用标签映射并确保存回列
train_data["label_L1"] = train_data["label_L1"].map(label_L1_mapping)
train_data["label_L2"] = train_data["label_L2"].map(label_L2_mapping)

# 移除 'label_L2' 并保留 'label_L1'
# train_label_L1, train_label_L2 = train_data["label_L1"], train_data.pop("label_L2")

# 移除 'label_L2' 和 'label_L1'
train_label_L1, train_label_L2 = train_data.pop("label_L1"), train_data.pop("label_L2")


# 使用 StandardScaler 对数据进行缩放
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
# Save the scaler after training
scaler_save_path = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/scaler_reduced_45features.joblib'
joblib.dump(scaler, scaler_save_path)