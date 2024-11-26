import matplotlib.pyplot as plt
import numpy as np

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
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

# # Plot for CIC IoT Dataset
# ax1.bar(x - 0.2, baseline_iot, width=0.4, label='Baseline', color='blue')
# ax1.bar(x + 0.2, lnn_iot, width=0.4, label='LNN', color='red')
# ax1.set_ylim(60, 90)
# ax1.set_ylabel('Scores (%)')
# ax1.set_title('(a) Baseline vs LNN - CIC IoT Dataset (4-11 Categories)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(metrics)
# ax1.legend()

# # Plot for CIC IoMT Dataset
# ax2.bar(x - 0.2, baseline_iomt, width=0.4, label='Baseline', color='blue')
# ax2.bar(x + 0.2, lnn_iomt, width=0.4, label='LNN', color='red')
# ax2.set_ylim(60, 90)
# ax2.set_ylabel('Scores (%)')
# ax2.set_title('(b) Baseline vs LNN - CIC IoMT Dataset (3-9 Categories)')
# ax2.set_xticks(x)
# ax2.set_xticklabels(metrics)
# ax2.legend()

# plt.xlabel('Metrics')
# plt.tight_layout()
# plt.show()
# # Save the plot as a PNG file
# fig.savefig("baseline_vs_lnn_comparison.png", format="png", dpi=300)
# plt.close(fig)  # Close the figure to free memory

# # File path saved
# "baseline_vs_lnn_comparison.png"
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
dataset_sizes = ['5k', '10k', '30k']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Performance metrics for each dataset size
baseline_5k = [63, 61, 61, 60]
lnn_5k = [68, 64, 65, 64]

baseline_10k = [81, 79, 75, 75]
lnn_10k = [85, 83, 84, 82]

baseline_30k = [84, 83, 81, 81.5]
lnn_30k = [84, 84, 83, 81]

# Plot settings
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0.3})
data_sizes = [baseline_5k, lnn_5k, baseline_10k, lnn_10k, baseline_30k, lnn_30k]
titles = ['Dataset Size: 5k', 'Dataset Size: 10k', 'Dataset Size: 30k']

# Create plots for each dataset size
for i, ax in enumerate(axes):
    x = np.arange(len(metrics))
    baseline = data_sizes[i * 2]
    lnn = data_sizes[i * 2 + 1]
    
    ax.bar(x - 0.2, baseline, width=0.4, label='Baseline', color='blue')
    ax.bar(x + 0.2, lnn, width=0.4, label='LNN', color='red')
    ax.set_ylim(50, 90)
    ax.set_ylabel('Scores (%)', fontsize=10)
    ax.set_title(titles[i], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)

plt.xlabel('Metrics', fontsize=12)

# Adjust layout to minimize margins
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
fig.savefig("performance_margin_adjusted.png", format="png", dpi=300, bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory
