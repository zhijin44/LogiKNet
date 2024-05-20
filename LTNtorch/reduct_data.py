import pandas as pd

# 读取训练集和测试集文件
processed_train_file = '../CIC_IoMT/19classes/processed_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/processed_test_data.csv'
# processed_train_file = '../CIC_IOT/0.1percent_8classes.csv'
# processed_train_file = '../CIC_IoMT/19classes/processed_train_data.csv'
# processed_test_file = '../CIC_IoMT/19classes/processed_test_data.csv'
# processed_train_file = '../CIC_IoMT/6classes/processed_train_data_6classes.csv'
# processed_test_file = '../CIC_IoMT/6classes/processed_test_data_6classes.csv'
# processed_train_file = '../CIC_IoMT/6classes/reduce_6classes_train.csv'
# processed_test_file = '../CIC_IoMT/6classes/reduce_6classes_test.csv'

# 加载数据
train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)

# 输出数据信息
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


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


print_label_counts(train_data)
print_label_counts(test_data)


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
