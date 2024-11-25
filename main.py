import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))

# CIC IoT Dataset
baseline_iot = [77, 76, 75, 73]
lnn_iot = [81, 80, 81, 78]

# CIC IoMT Dataset
baseline_iomt = [81, 79, 75, 75]
lnn_iomt = [85, 83, 84, 82]

# Plot settings
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

# Plot for CIC IoT Dataset
ax1.bar(x - 0.2, baseline_iot, width=0.4, label='Baseline', color='blue')
ax1.bar(x + 0.2, lnn_iot, width=0.4, label='LNN', color='red')
ax1.set_ylim(60, 90)
ax1.set_ylabel('Scores (%)')
ax1.set_title('Baseline vs LNN - CIC IoT Dataset (4-11 Categories)')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

# Plot for CIC IoMT Dataset
ax2.bar(x - 0.2, baseline_iomt, width=0.4, label='Baseline', color='blue')
ax2.bar(x + 0.2, lnn_iomt, width=0.4, label='LNN', color='red')
ax2.set_ylim(60, 90)
ax2.set_ylabel('Scores (%)')
ax2.set_title('Baseline vs LNN - CIC IoMT Dataset (3-9 Categories)')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()

plt.xlabel('Metrics')
plt.tight_layout()
plt.show()
# Save the plot as a PNG file
fig.savefig("baseline_vs_lnn_comparison.png", format="png", dpi=300)
plt.close(fig)  # Close the figure to free memory

# File path saved
"baseline_vs_lnn_comparison.png"
