import pandas as pd

# 读取训练集和测试集文件
# processed_train_file = '../CIC_IoMT/19classes/reduced_train_data.csv'
# processed_test_file = '../CIC_IoMT/19classes/reduced_test_data.csv'
# processed_train_file = '../CIC_IoMT/19classes/filtered_train_data.csv'
# processed_test_file = '../CIC_IoMT/19classes/filtered_test_data.csv'
# processed_train_file = '../CIC_IOT/0.1percent_8classes.csv'
processed_train_file = '../CIC_IoMT/19classes/processed_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/processed_test_data.csv'


# processed_train_file = '../CIC_IoMT/6classes/processed_train_data_6classes.csv'
# processed_test_file = '../CIC_IoMT/6classes/processed_test_data_6classes.csv'
# processed_train_file = '../CIC_IoMT/6classes/reduce_6classes_train.csv'
# processed_test_file = '../CIC_IoMT/6classes/reduce_6classes_test.csv'

# 加载数据
# train_data = pd.read_csv(processed_train_file)
# test_data = pd.read_csv(processed_test_file)

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


# print_label_counts(train_data)
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

        print("Label_L1 Counts:")
        for label, count in label_L1_counts.items():
            print(f"Label_L1 {label}: {count} entries")

        print("\nLabel_L2 Counts:")
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
def reduce_instances(df, labels_to_reduce, reduction_fraction=0.01):
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
# reduced_train_data = reduce_instances(train_data, ["TCP_IP-DDOS", "TCP_IP-DOS"], 0.01)
# reduced_test_data = reduce_instances(test_data, ["TCP_IP-DDOS", "TCP_IP-DOS"], 0.01)


###############################################################################

def create_reduced_dataset(data, n_samples=64):
    # Get the unique labels
    labels = data['label'].unique()  # Assuming 'label' is the name of your label column

    reduced_data = pd.DataFrame()  # Initialize an empty DataFrame to store the reduced dataset

    for label in labels:
        # For each label, randomly select n_samples
        sampled_data = data[data['label'] == label].sample(n=n_samples, random_state=42)
        reduced_data = pd.concat([reduced_data, sampled_data], ignore_index=True)

    return reduced_data


# # Create reduced datasets
# reduced_train_data = create_reduced_dataset(train_data, 15000)
# reduced_test_data = create_reduced_dataset(test_data, 1700)
#
# print("Training data shape:", reduced_train_data.shape)
# print("Test data shape:", reduced_test_data.shape)
#
# reduced_train_data.to_csv("../CIC_IoMT/6classes/6classes_15k_train.csv", index=False)
# reduced_test_data.to_csv("../CIC_IoMT/6classes/6classes_1700_test.csv", index=False)

#####################################################################################
############# Part of the data from IoMT (benign & MQTT)#####################

# # 要提取的六类标签
# selected_labels = [
#     "Benign",
#     "MQTT-DDoS-Connect_Flood",
#     "MQTT-DoS-Publish_Flood",
#     "MQTT-DDoS-Publish_Flood",
#     "MQTT-DoS-Connect_Flood",
#     "MQTT-Malformed_Data"
# ]

# 提取对应标签的数据
# filtered_train_data = train_data[train_data['label'].isin(selected_labels)]
# filtered_test_data = test_data[test_data['label'].isin(selected_labels)]

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
def reduce_data(train_file, test_file, selected_labels, max_samples=20000):
    # 读取训练和测试数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 初始化空的DataFrame用于存储缩减后的数据
    reduced_train_data = pd.DataFrame()
    reduced_test_data = pd.DataFrame()

    # 遍历每个标签并缩减数据
    for label in selected_labels:
        train_subset = train_data[train_data['label'] == label]
        test_subset = test_data[test_data['label'] == label]

        if len(train_subset) > max_samples:
            train_subset = train_subset.head(max_samples)
        if len(test_subset) > max_samples:
            test_subset = test_subset.head(max_samples)

        reduced_train_data = pd.concat([reduced_train_data, train_subset], ignore_index=True)
        reduced_test_data = pd.concat([reduced_test_data, test_subset], ignore_index=True)

    # 将非selected_labels的数据也加入到缩减后的数据集中
    other_train_data = train_data[~train_data['label'].isin(selected_labels)]
    other_test_data = test_data[~test_data['label'].isin(selected_labels)]

    reduced_train_data = pd.concat([reduced_train_data, other_train_data], ignore_index=True)
    reduced_test_data = pd.concat([reduced_test_data, other_test_data], ignore_index=True)

    return reduced_train_data, reduced_test_data


def extend_labels(train_data, test_data, label_L1_mapping):
    # 映射新的label_L1
    def map_label_L1(label):
        for l1 in label_L1_mapping:
            if l1.replace("-", "").lower() in label.lower().replace("-", ""):
                return l1
        return 'Other'

    train_data['label_L1'] = train_data['label'].apply(map_label_L1)
    test_data['label_L1'] = test_data['label'].apply(map_label_L1)

    # 修改 'label' 列名为 'label_L2'
    train_data.rename(columns={'label': 'label_L2'}, inplace=True)
    test_data.rename(columns={'label': 'label_L2'}, inplace=True)

    # 生成新的文件名
    extended_train_file = '../CIC_IoMT/19classes/reduced_train_data.csv'
    extended_test_file = '../CIC_IoMT/19classes/reduced_test_data.csv'

    # 保存扩展后的数据集到新的CSV文件
    train_data.to_csv(extended_train_file, index=False)
    test_data.to_csv(extended_test_file, index=False)

    print(f"Extended train data saved to: {extended_train_file}")
    print(f"Extended test data saved to: {extended_test_file}")


# 使用示例
selected_labels = [
    "TCP_IP-DDoS-UDP",
    "TCP_IP-DDoS-ICMP",
    "TCP_IP-DDoS-TCP",
    "TCP_IP-DDoS-SYN",
    "TCP_IP-DoS-UDP",
    "TCP_IP-DoS-SYN",
    "TCP_IP-DoS-ICMP",
    "TCP_IP-DoS-TCP"
]
label_L1_mapping = [
    "ARP_Spoofing", "Benign", "MQTT", "Recon", "TCP_IP-DDOS", "TCP_IP-DOS"
]

# 先进行数据缩减
reduced_train_data, reduced_test_data = reduce_data(processed_train_file, processed_test_file, selected_labels)

# 然后扩展标签并保存数据
extend_labels(reduced_train_data, reduced_test_data, label_L1_mapping)

extended_train_data = pd.read_csv('../CIC_IoMT/19classes/reduced_train_data.csv')
extended_test_data = pd.read_csv('../CIC_IoMT/19classes/reduced_test_data.csv')
# 打印扩展标签的统计信息
print("Train Data:")
print_extended_label_counts(extended_train_data)
print("\nTest Data:")
print_extended_label_counts(extended_test_data)
