import pandas as pd
from sklearn.model_selection import train_test_split

label_L1_mapping = [
    "ARP_Spoofing", "Benign", "MQTT", "Recon", "TCP_IP-DDOS", "TCP_IP-DOS"
]

# 读取训练集和测试集文件
processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_med_train.csv'
processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_med_test.csv'
# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/processed_train_data.csv'
# processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/processed_test_data.csv'


# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/processed_train_data_6classes.csv'
# processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/processed_test_data_6classes.csv'
# processed_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_100k_train.csv'
# processed_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_10k_test.csv'

# 加载数据
train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 输出数据信息
# print("Training data shape:", train_data.shape)
# print("Test data shape:", test_data.shape)


def print_label_counts(df):
    """
    Print the counts of each label in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data and labels.

    Returns:
    None
    """
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("Label Counts:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} entries")

    else:
        print("The DataFrame does not have a 'label' column.")

    # if 'label_L1' in df.columns:
    #     label_counts = df['label_L1'].value_counts()
    #     print("Label Counts:")
    #     for label, count in label_counts.items():
    #         print(f"Label {label}: {count} entries")
    #
    # else:
    #     print("The DataFrame does not have a 'label' column.")

# print("Training data shape:", train_data.shape)
# print_label_counts(train_data)
# print("Test data shape:", test_data.shape)
# print_label_counts(test_data)


def print_extended_label_counts(df):
    """
    Print the counts of each label in the given DataFrame for label_L1 and label_L2.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data and labels.

    Returns:
    None
    """
    if 'label_L1' in df.columns and 'label_L2' in df.columns:
        label_L1_counts = df['label_L1'].value_counts()
        label_L2_counts = df['label_L2'].value_counts()

        print("label_L1 Counts:")
        for label, count in label_L1_counts.items():
            print(f"Label_L1 {label}: {count} entries")

        print("\nlabel_L2 Counts:")
        for label, count in label_L2_counts.items():
            print(f"Label_L2 {label}: {count} entries")

    else:
        missing_columns = []
        if 'label_L1' not in df.columns:
            missing_columns.append('label_L1')
        if 'label_L2' not in df.columns:
            missing_columns.append('label_L2')
        print(f"The DataFrame does not have the following column(s): {', '.join(missing_columns)}")


# extended_train_data = pd.read_csv(processed_train_file)
# extended_test_data = pd.read_csv(processed_test_file)
# # 打印扩展标签的统计信息
# print("Train Data:")
# print_extended_label_counts(extended_train_data)
# print("\nTest Data:")
# print_extended_label_counts(extended_test_data)


#########################################################################
def reduce_instances(df, labels_to_reduce, reduction_fraction=0.5):
    """
    减少特定标签类别的实例至原数据的一定比例。

    :param df: 原始的DataFrame。
    :param labels_to_reduce: 需要减少实例的标签列表。
    :param reduction_fraction: 保留的数据比例。
    :return: 减少特定标签实例后的新DataFrame。
    """
    # 分离出需要减少实例的数据和其它数据
    df_to_reduce = df[df['label'].isin(labels_to_reduce)]
    df_remaining = df[~df['label'].isin(labels_to_reduce)]

    # 对每个需要减少实例的标签进行抽样
    df_reduced = pd.DataFrame()  # 初始化一个空的DataFrame来存储抽样后的数据
    for label in labels_to_reduce:
        df_filtered = df_to_reduce[df_to_reduce['label'] == label]
        # 进行抽样
        df_sampled = df_filtered.sample(frac=reduction_fraction, random_state=1)
        df_reduced = pd.concat([df_reduced, df_sampled], axis=0)

    # 将减少后的数据与未被过滤的数据合并
    df_final = pd.concat([df_reduced, df_remaining], axis=0)

    return df_final


# 调用函数
# reduced_train_data = reduce_instances(train_data, label_L1_mapping, 0.3)
# reduced_test_data = reduce_instances(test_data, label_L1_mapping, 0.3)

# print("Training data shape:", reduced_train_data.shape)
# print_label_counts(reduced_train_data)
# print("Test data shape:", reduced_test_data.shape)
# print_label_counts(reduced_test_data)

# reduced_train_data.to_csv("/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_30k_train.csv", index=False)
# reduced_test_data.to_csv("/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_3k_test.csv", index=False)


###############################################################################

def reduce_to_n_samples(data, n_samples=64):
    # Get the unique labels
    labels = data['label'].unique()  # Assuming 'label' is the name of your label column

    reduced_data = pd.DataFrame()  # Initialize an empty DataFrame to store the reduced dataset

    for label in labels:
        # For each label, randomly select n_samples
        sampled_data = data[data['label'] == label].sample(n=n_samples, random_state=42)
        reduced_data = pd.concat([reduced_data, sampled_data], ignore_index=True)

    return reduced_data


# # Create reduced datasets
# reduced_train_data = reduce_to_n_samples(train_data, 16047)
# reduced_test_data = reduce_to_n_samples(test_data, 1744)

# print("Training data shape:", reduced_train_data.shape)
# print("Test data shape:", reduced_test_data.shape)

# reduced_train_data.to_csv("/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_100k_train.csv", index=False)
# reduced_test_data.to_csv("/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_10k_test.csv", index=False)

#####################################################################################
############# Part of the data from IoMT (benign & MQTT)#####################

# 要提取的六类标签
selected_labels = [
    "Benign",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
    "MQTT-Malformed_Data"
]

# 提取对应标签的数据
filtered_train_data = train_data[train_data['label_L2'].isin(selected_labels)]
filtered_test_data = test_data[test_data['label_L2'].isin(selected_labels)]

print_extended_label_counts(filtered_train_data)
print_extended_label_counts(filtered_test_data)
# 导出新的数据集
filtered_train_data.to_csv('/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_data.csv', index=False)
filtered_test_data.to_csv('/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_data.csv', index=False)


# 新建feature列label_L1和label_L2
def create_labels(row):
    label = row['label']
    if label == "Benign":
        return "Benign", "Benign"
    elif "MQTT" in label:
        parts = label.split('-', 1)
        return parts[0], parts[1]
    else:
        return label, label


# filtered_train_data[['label_L1', 'label_L2']] = filtered_train_data.apply(create_labels, axis=1, result_type="expand")
# filtered_test_data[['label_L1', 'label_L2']] = filtered_test_data.apply(create_labels, axis=1, result_type="expand")
#
# # 删除原始的Label列
# filtered_train_data.drop(columns=['label'], inplace=True)
# filtered_test_data.drop(columns=['label'], inplace=True)
#
# # 导出新的数据集
# filtered_train_data.to_csv('../CIC_IoMT/19classes/filtered_train_data.csv', index=False)
# filtered_test_data.to_csv('../CIC_IoMT/19classes/filtered_test_data.csv', index=False)
#
# # 查看数据集的结构
# print(filtered_train_data.head())
# print(filtered_test_data.head())
#
# print("数据处理完成并导出新的数据集。")

#####################################################################################
############# reduce the huge data from IoMT (Dos & DDos)#####################
def reduce_data(train_file, selected_labels, max_samples):
    # 读取训练和测试数据
    train_data = pd.read_csv(train_file)

    # 初始化空的DataFrame用于存储缩减后的数据
    reduced_train_data = pd.DataFrame()

    # 遍历每个标签并缩减数据
    for label in selected_labels:
        train_subset = train_data[train_data['label'] == label]

        if len(train_subset) > max_samples:
            train_subset = train_subset.head(max_samples)

        reduced_train_data = pd.concat([reduced_train_data, train_subset], ignore_index=True)

    # 将非selected_labels的数据也加入到缩减后的数据集中
    other_train_data = train_data[~train_data['label'].isin(selected_labels)]

    reduced_train_data = pd.concat([reduced_train_data, other_train_data], ignore_index=True)

    return reduced_train_data


def extend_labels(train_data, label_L1_mapping):
    # 映射新的label_L1
    def map_label_L1(label):
        for l1 in label_L1_mapping:
            if l1.replace("-", "").lower() in label.lower().replace("-", ""):
                return l1
        return 'Other'

    train_data['label_L1'] = train_data['label'].apply(map_label_L1)

    # 修改 'label' 列名为 'label_L2'
    train_data.rename(columns={'label': 'label_L2'}, inplace=True)

    # Modify specific values in the 'label_L2' column
    # train_data['label_L2'] = train_data['label_L2'].replace({"Benign": "benign", "ARP_Spoofing": "arp_spoofing"})

    return train_data


# 使用示例
selected_labels = [
    "TCP_IP-DDoS-UDP",
    "TCP_IP-DDoS-ICMP",
    "TCP_IP-DDoS-TCP",
    "TCP_IP-DDoS-SYN",
    "TCP_IP-DoS-UDP",
    "TCP_IP-DoS-SYN",
    "TCP_IP-DoS-ICMP",
    "TCP_IP-DoS-TCP",
    "Recon-Port_Scan",
    "Recon-OS_Scan",
    "ARP_Spoofing",
    "Benign",
    "MQTT-DDoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood",
    "MQTT-DDoS-Publish_Flood",
    "MQTT-DoS-Connect_Flood",
  ]

# extended_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_train.csv'
# extended_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/19classes_test.csv'

# # 先进行数据缩减
# reduced_train_data = reduce_data(processed_train_file, selected_labels, 10000)
# reduced_test_data = reduce_data(processed_test_file, selected_labels, 10000)

# # 保存扩展后的数据集到新的CSV文件
# train_data.to_csv(extended_train_file, index=False)
# test_data.to_csv(extended_test_file, index=False)

# print(f"Extended train data saved to: {extended_train_file}")

#####################################################################################
########## for 19 classes, combining train and test together and divide later#####
# # Combine the datasets by concatenating rows
# combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
# # print_label_counts(combined_data)

# # Step 1: Calculate total entries per class
# label_counts = combined_data['label'].value_counts()
# # Step 2: Define threshold for minority vs majority classes
# ## minority_threshold = 10000
# minority_threshold = 1000
# # Step 3: Separate minority and majority classes
# minority_classes = label_counts[label_counts <= minority_threshold].index
# majority_classes = label_counts[label_counts > minority_threshold].index
# # Step 4: Keep all entries from minority classes
# minority_data = combined_data[combined_data['label'].isin(minority_classes)]
# majority_data = combined_data[combined_data['label'].isin(majority_classes)]
# ## majority_data = reduce_to_n_samples(majority_data, 15000)
# majority_data = reduce_to_n_samples(majority_data, 1000)

# combined_data = pd.concat([minority_data, majority_data], axis=0, ignore_index=True)
# # print_label_counts(combined_data)

# extended_data = extend_labels(combined_data, label_L1_mapping)
# print_extended_label_counts(extended_data)

# train_data, test_data = train_test_split(extended_data, test_size=0.1, random_state=42, stratify=extended_data['label_L2'])
# extended_train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_100k_train.csv'
# extended_test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/6classes/6classes_10k_test.csv'
# train_data.to_csv(extended_train_file, index=False)
# test_data.to_csv(extended_test_file, index=False)
# print("Training and testing datasets saved successfully.")


