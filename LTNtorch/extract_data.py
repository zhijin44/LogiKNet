import pandas as pd
from sklearn.model_selection import train_test_split


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

def extract_selected_labels(df, selected_labels):
    """
    Extract rows from the DataFrame where 'label_L2' is in the selected labels list.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    selected_labels (list): List of label_L2 values to extract.

    Returns:
    pandas.DataFrame: Filtered DataFrame containing only rows with label_L2 in selected_labels.
    """
    # Ensure no typos or whitespace issues by stripping whitespaces from 'label_L2'
    df['label_L2'] = df['label_L2'].str.strip()
    
    # Filter rows based on the selected labels for label_L2
    filtered_data = df[df['label_L2'].isin(selected_labels)]
    
    return filtered_data

def reduce_data_by_label(df, label_column, n):
    """
    Reduce the number of rows for each unique label in the specified column to n rows.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to reduce.
    label_column (str): The name of the column containing the labels.
    n (int): The maximum number of rows to keep for each label.

    Returns:
    pandas.DataFrame: A reduced DataFrame with at most n rows per label.
    """
    # Create an empty DataFrame to store the reduced data
    reduced_data = pd.DataFrame()

    # Loop over each unique label in the specified column
    for label in df[label_column].unique():
        # Extract all rows for the current label
        label_data = df[df[label_column] == label]
        
        # Sample n rows or all rows if fewer than n exist
        reduced_label_data = label_data.sample(n=min(len(label_data), n), random_state=42)
        
        # Append the reduced data for this label to the result
        reduced_data = pd.concat([reduced_data, reduced_label_data], ignore_index=True)
    
    return reduced_data

def split_and_save_data(df, train_ratio=0.9, train_file='train_data.csv', test_file='test_data.csv'):
    """
    Split the DataFrame into training and testing sets based on a specified ratio, then save to CSV.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.
    train_ratio (float): The proportion of the data to include in the train set.
    train_file (str): File path to save the train set.
    test_file (str): File path to save the test set.

    Returns:
    tuple: (train_data, test_data) DataFrames after splitting.
    """
    # Split the data
    train_data, test_data = train_test_split(df, test_size=1 - train_ratio, random_state=42, stratify=df['label_L2'])

    # Save to CSV files
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training data saved to {train_file} with shape {train_data.shape}")
    print(f"Testing data saved to {test_file} with shape {test_data.shape}")
    
    return train_data, test_data

def reduce_data_custom(df, label_column, n, m, specific_label='benign'):
    """
    Reduce the number of rows for each unique label in the specified column.
    Retain at most n rows for all labels except a specific label, for which m rows are retained.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to reduce.
    label_column (str): The name of the column containing the labels.
    n (int): The maximum number of rows to keep for labels other than the specific label.
    m (int): The maximum number of rows to keep for the specific label.
    specific_label (str): The label for which m rows will be retained.

    Returns:
    pandas.DataFrame: A reduced DataFrame with at most n rows per label (except m rows for specific_label).
    """
    # Create an empty DataFrame to store the reduced data
    reduced_data = pd.DataFrame()

    # Loop over each unique label in the specified column
    for label in df[label_column].unique():
        # Extract all rows for the current label
        label_data = df[df[label_column] == label]

        # Determine the number of rows to sample
        if label == specific_label:
            max_rows = m
        else:
            max_rows = n

        # Sample rows or take all rows if fewer than the limit
        reduced_label_data = label_data.sample(n=min(len(label_data), max_rows), random_state=42)

        # Append the reduced data for this label to the result
        reduced_data = pd.concat([reduced_data, reduced_label_data], ignore_index=True)
    
    return reduced_data


train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_train_data.csv'
test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/reduced_test_data.csv'

# 加载数据
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Combine the datasets by concatenating rows
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
print_extended_label_counts(combined_data)

selected_labels = [
    'benign',
    'MQTT-DDoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood',
    'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood',
    'MQTT-Malformed_Data',
    'Recon-Port_Scan',
    'Recon-OS_Scan',
    'arp_spoofing'
]

# Apply the function to the combined data
filtered_data = extract_selected_labels(combined_data, selected_labels)

# Print the shape and label counts for verification
print("Filtered Data Shape:", filtered_data.shape)
print_extended_label_counts(filtered_data)

n = 10000  # Maximum rows for all labels except 'benign'
m = 30000  # Maximum rows for 'benign'

# reduced_data = reduce_data_by_label(filtered_data, label_column='label_L2', n=n)
# # Print the shape and label counts for verification
# print("Reduced Data Shape:", reduced_data.shape)
# print_extended_label_counts(reduced_data)

reduced_data_custom = reduce_data_custom(filtered_data, label_column='label_L2', n=n, m=m, specific_label='benign')
# Print the shape and label counts for verification
print("Custom Reduced Data Shape:", reduced_data_custom.shape)
print_extended_label_counts(reduced_data_custom)




train_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_train_4_9.csv'
test_file = '/home/zyang44/Github/baseline_cicIOT/CIC_IoMT/19classes/filtered_test_4_9.csv'
# train_data, test_data = split_and_save_data(reduced_data, train_ratio=0.9, train_file=train_file, test_file=test_file)
train_data, test_data = split_and_save_data(reduced_data_custom, train_ratio=0.9, train_file=train_file, test_file=test_file)

