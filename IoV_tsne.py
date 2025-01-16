import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the file path
file_path = '/home/zyang44/Github/baseline_cicIOT/IoV_power_tiny.csv'
# Load the dataset
df = pd.read_csv(file_path)
print(df.head())

# Encode categorical columns (e.g., 'State' and 'Attack' if necessary)
attack_mapping = {'syn-flood': 0, 'tcp-flood': 1, 'none': 2, 'cryptojacking': 3, 'syn-stealth': 4, 'vuln-scan': 5, 'Backdoor': 6}
state_mapping = {'idle': 0, 'charging': 1}
df['State'] = df['State'].map(state_mapping)
df['Attack'] = df['Attack'].map(attack_mapping)

# Separate features and labels
df.pop('State')
print(df.head())
X = df.drop(columns=["Attack"])  # Replace 'Attack' if your label column is named differently
y = df["Attack"]  # Replace 'Attack' if your label column is named differently

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#####################################################################################
# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Convert the embedded points into a DataFrame
tsne_df = pd.DataFrame(X_embedded, columns=["Component 1", "Component 2"])
# Add the actual labels to the DataFrame
reverse_attack_mapping = {v: k for k, v in attack_mapping.items()}
tsne_df["Label"] = y.map(reverse_attack_mapping)

# Plot t-SNE results
plt.figure(figsize=(10, 6))
for label in tsne_df["Label"].unique():
    subset = tsne_df[tsne_df["Label"] == label]
    plt.scatter(subset["Component 1"], subset["Component 2"], label=label, alpha=0.6)

plt.title("t-SNE Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IoV-tsne", dpi=300)

#####################################################################################
# Perform t-SNE with 3 components
tsne = TSNE(n_components=3, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Convert the embedded points into a DataFrame
tsne_df = pd.DataFrame(X_embedded, columns=["Component 1", "Component 2", "Component 3"])
# Add the actual labels to the DataFrame
reverse_attack_mapping = {v: k for k, v in attack_mapping.items()}
tsne_df["Label"] = y.map(reverse_attack_mapping)

# Plot t-SNE results in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in tsne_df["Label"].unique():
    subset = tsne_df[tsne_df["Label"] == label]
    ax.scatter(
        subset["Component 1"], 
        subset["Component 2"], 
        subset["Component 3"], 
        label=label, 
        alpha=0.6
    )

ax.set_title("3D t-SNE Visualization")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IoV-tsne-3d", dpi=300)