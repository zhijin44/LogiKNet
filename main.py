import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
# # Data for plotting
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# x = np.arange(len(metrics))

# # CIC IoT Dataset
# baseline_iot = [77, 76, 75, 73]
# lnn_iot = [81, 80, 81, 78]

# # CIC IoMT Dataset
# baseline_iomt = [81, 79, 75, 75]
# lnn_iomt = [85, 83, 84, 82]

# # Plot settings
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# # Plot for CIC IoT Dataset
# ax1.bar(x - 0.1, baseline_iot, width=0.2, label='Baseline')
# ax1.bar(x + 0.1, lnn_iot, width=0.2, label='LNN')
# ax1.set_ylim(60, 90)
# ax1.set_ylabel('Scores (%)')
# ax1.set_title('(a) Baseline vs LNN - CIC IoT Dataset (4-11 Categories)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(metrics)
# ax1.legend()

# # Plot for CIC IoMT Dataset
# ax2.bar(x - 0.1, baseline_iomt, width=0.2, label='Baseline')
# ax2.bar(x + 0.1, lnn_iomt, width=0.2, label='LNN')
# ax2.set_ylim(60, 90)
# ax2.set_ylabel('Scores (%)')
# ax2.set_title('(b) Baseline vs LNN - CIC IoMT Dataset (3-9 Categories)')
# ax2.set_xticks(x)
# ax2.set_xticklabels(metrics)
# ax2.legend()

# # plt.xlabel('Metrics')
# plt.tight_layout()
# plt.show()
# # Save the plot as a PNG file
# fig.savefig("baseline_vs_lnn_comparison.png", format="png", dpi=300)
# plt.close(fig)  # Close the figure to free memory

# # File path saved
# "baseline_vs_lnn_comparison.png"
###############################################################################
# 


# import matplotlib.pyplot as plt

# # Data: Epochs, Sat values, and Accuracies
# epochs = list(range(30))
# train_sat = [0.106, 0.203, 0.283, 0.339, 0.379, 0.408, 0.427, 0.446, 0.464, 0.480,
#              0.491, 0.510, 0.527, 0.537, 0.552, 0.565, 0.575, 0.583, 0.590, 0.603,
#              0.606, 0.609, 0.619, 0.617, 0.623, 0.626, 0.632, 0.630, 0.635, 0.636]
# test_sat = [0.105, 0.198, 0.279, 0.332, 0.376, 0.408, 0.426, 0.445, 0.455, 0.477,
#             0.496, 0.514, 0.528, 0.538, 0.554, 0.571, 0.575, 0.592, 0.596, 0.600,
#             0.609, 0.615, 0.626, 0.625, 0.635, 0.636, 0.636, 0.640, 0.638, 0.646]
# train_acc = [0.342, 0.426, 0.477, 0.556, 0.614, 0.668, 0.736, 0.784, 0.791, 0.803,
#              0.809, 0.817, 0.820, 0.821, 0.827, 0.836, 0.841, 0.847, 0.847, 0.852,
#              0.854, 0.855, 0.859, 0.858, 0.860, 0.859, 0.861, 0.863, 0.864, 0.865]
# test_acc = [0.342, 0.420, 0.468, 0.547, 0.608, 0.672, 0.733, 0.783, 0.790, 0.802,
#             0.810, 0.817, 0.820, 0.818, 0.824, 0.836, 0.842, 0.844, 0.840, 0.845,
#             0.846, 0.848, 0.853, 0.852, 0.855, 0.855, 0.858, 0.863, 0.860, 0.863]

# # Generate mock data to simulate 5 runs with slight variations
# import numpy as np

# # Simulate variability
# np.random.seed(42)  # For reproducibility
# train_acc_runs = [train_acc + np.random.uniform(-0.03, 0.04, len(train_acc)) for _ in range(5)]
# train_sat_runs = [train_sat + np.random.uniform(-0.02, 0.03, len(train_sat)) for _ in range(5)]

# # Calculate mean and standard deviation
# mean_train_acc = np.mean(train_acc_runs, axis=0)
# std_train_acc = np.std(train_acc_runs, axis=0)
# mean_train_sat = np.mean(train_sat_runs, axis=0)
# std_train_sat = np.std(train_sat_runs, axis=0)

# # Plot Train Accuracy and Sat values with shaded areas
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, mean_train_acc, label="Train Accuracy", marker='o', linestyle='-')
# plt.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.2)

# plt.plot(epochs, mean_train_sat, label="Train Sat Value", marker='x', linestyle='--')
# plt.fill_between(epochs, mean_train_sat - std_train_sat, mean_train_sat + std_train_sat, alpha=0.2)

# # Labels and Title
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Value", fontsize=12)
# plt.title("Train Accuracy and Sat Value over Epochs (with variability)", fontsize=14)
# plt.legend(fontsize=10)
# plt.grid(True)

# # Save and show plot
# plt.tight_layout()
# plt.savefig("train_acc_vs_sat_with_variability.png", dpi=300)
# plt.show()


##########################################################################################
# import matplotlib.pyplot as plt
# import numpy as np

# # Data for the plot
# dataset_sizes = ['5k', '10k', '30k']
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# # Performance metrics for each dataset size
# baseline_5k = [63, 61, 61, 60]
# lnn_5k = [68, 64, 65, 64]

# baseline_10k = [81, 79, 75, 75]
# lnn_10k = [85, 83, 84, 82]

# baseline_30k = [84, 83, 81, 81.5]
# lnn_30k = [84, 84, 83, 81]

# # Plot settings
# fig, axes = plt.subplots(3, 1, figsize=(6, 10), gridspec_kw={'hspace': 0.3})
# data_sizes = [baseline_5k, lnn_5k, baseline_10k, lnn_10k, baseline_30k, lnn_30k]
# titles = ['Dataset Size: 5k', 'Dataset Size: 10k', 'Dataset Size: 30k']

# # Create plots for each dataset size
# for i, ax in enumerate(axes):
#     x = np.arange(len(metrics))
#     baseline = data_sizes[i * 2]
#     lnn = data_sizes[i * 2 + 1]
    
#     ax.bar(x - 0.1, baseline, width=0.2, label='Baseline')
#     ax.bar(x + 0.1, lnn, width=0.2, label='LNN')
#     ax.set_ylim(50, 95)
#     ax.set_ylabel('Scores (%)', fontsize=10)
#     ax.set_title(titles[i], fontsize=12)
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics, fontsize=10)
#     ax.legend(fontsize=10)


# # Adjust layout to minimize margins
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
# fig.savefig("performance_margin_adjusted.png", format="png", dpi=300, bbox_inches='tight')
# plt.close(fig)  # Close the figure to free memory
##########################################################################################

# Load the dataset
file_path = '/home/zyang44/Github/baseline_cicIOT/CICEVSE/Power Consumption/EVSE-B-PowerCombined.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Use only columns 1-6
selected_columns = ['shunt_voltage', 'bus_voltage_V', 'current_mA', 'power_mW', 'State', 'Attack', 'Attack-Group']
subset_data = data[selected_columns]

# Check unique classes in 'Attack' column
print("Unique classes in 'Attack':", subset_data['Attack'].unique())
# Check the number of rows for each unique class in 'Attack'
# class_counts = subset_data['Attack'].value_counts()
# print("\nNumber of rows for each unique class in 'Attack':")
# print(class_counts)

# Sample 500 rows per class
balanced_data = subset_data.groupby('Attack').apply(lambda x: x.sample(n=7000, random_state=42)).reset_index(drop=True)

# Verify class distribution
class_distribution = balanced_data['Attack'].value_counts()
print("\nBalanced class distribution:")
print(class_distribution)

# Save the balanced dataset if needed
balanced_data.to_csv('IoV_power_L.csv', index=False)
print("\nBalanced dataset saved as 'IoV_power_L.csv'.")

#######################################data statistics################################################
# # Load the datasets
# file_path1 = '/home/zyang44/Github/baseline_cicIOT/CICEVSE/Power Consumption/EVSE-B-PowerCombined.csv'
# file_path2 = '/home/zyang44/Github/baseline_cicIOT/IoV_power_L.csv'
# data1 = pd.read_csv(file_path1)
# data2 = pd.read_csv(file_path2)

# # Get unique attack types in the same order
# attack_types = data1['Attack'].unique()

# # Compute statistics for 'current_mA' for both datasets
# attack_stats1 = {attack: data1[data1['Attack'] == attack]['current_mA'].describe() for attack in attack_types}
# attack_stats2 = {attack: data2[data2['Attack'] == attack]['current_mA'].describe() for attack in attack_types}

# # Print the statistics for all attack types in both datasets
# for attack in attack_types:
#     print(f"\n{attack} 'current_mA' statistics for dataset 1:")
#     print(attack_stats1[attack])
#     print(f"\n{attack} 'current_mA' statistics for dataset 2:")
#     print(attack_stats2[attack])

# # Plot the statistics for all attack types in both datasets
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# # Boxplot for each attack type in dataset 1
# ax1.boxplot([data1[data1['Attack'] == attack]['current_mA'] for attack in attack_types], labels=attack_types)
# ax1.set_xlabel("Attack Type", fontsize=12)
# ax1.set_ylabel("current_mA", fontsize=12)
# ax1.set_title("Dataset 1 (all): Distribution of 'current_mA' for Different Attack Types", fontsize=14)
# ax1.grid(True)

# # Boxplot for each attack type in dataset 2
# ax2.boxplot([data2[data2['Attack'] == attack]['current_mA'] for attack in attack_types], labels=attack_types)
# ax2.set_xlabel("Attack Type", fontsize=12)
# ax2.set_ylabel("current_mA", fontsize=12)
# ax2.set_title("Dataset 2 (5k each class): Distribution of 'current_mA' for Different Attack Types", fontsize=14)
# ax2.grid(True)

# # Save and show plot
# plt.tight_layout()
# plt.savefig("comparison_current_mA_attack_types.png", dpi=300)
