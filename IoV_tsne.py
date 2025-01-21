import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the file path
file_path = '/home/zyang44/Github/baseline_cicIOT/IoV_power_L.csv'
# Load the dataset
selected_columns = ['shunt_voltage', 'bus_voltage_V', 'current_mA', 'power_mW', 'Attack']
df = pd.read_csv(file_path)[selected_columns]
print(df.head())

# Encode categorical columns (e.g., 'State' and 'Attack' if necessary)
# state_mapping = {'idle': 0, 'charging': 1}
# df['State'] = df['State'].map(state_mapping)
attack_mapping = {'syn-flood': 0, 'tcp-flood': 1, 'none': 2, 'cryptojacking': 3, 'syn-stealth': 4, 'vuln-scan': 5, 'Backdoor': 6}
df['Attack'] = df['Attack'].map(attack_mapping)

# Separate features and labels
X = df.drop(columns=["Attack"])  # Replace 'Attack' if your label column is named differently
y = df["Attack"]  # Replace 'Attack' if your label column is named differently

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#####################################################################################
# # Perform t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# X_embedded = tsne.fit_transform(X_scaled)

# # Convert the embedded points into a DataFrame
# tsne_df = pd.DataFrame(X_embedded, columns=["Component 1", "Component 2"])
# # Add the actual labels to the DataFrame
# reverse_attack_mapping = {v: k for k, v in attack_mapping.items()}
# tsne_df["Label"] = y.map(reverse_attack_mapping)

# # Plot t-SNE results
# plt.figure(figsize=(10, 6))
# for label in tsne_df["Label"].unique():
#     subset = tsne_df[tsne_df["Label"] == label]
#     plt.scatter(subset["Component 1"], subset["Component 2"], label=label, alpha=0.6)

# plt.title("t-SNE Visualization")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("IoV_tiny_tsne", dpi=300)

#####################################################################################
# # Perform t-SNE with 3 components
tsne = TSNE(n_components=3, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Convert the embedded points into a DataFrame
tsne_df = pd.DataFrame(X_embedded, columns=["Component 1", "Component 2", "Component 3"])
# Add the actual labels to the DataFrame
reverse_attack_mapping = {v: k for k, v in attack_mapping.items()}
tsne_df["Label"] = y.map(reverse_attack_mapping)

# # Plot t-SNE results in 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for label in tsne_df["Label"].unique():
#     subset = tsne_df[tsne_df["Label"] == label]
#     ax.scatter(
#         subset["Component 1"], 
#         subset["Component 2"], 
#         subset["Component 3"], 
#         label=label, 
#         alpha=0.6
#     )

# ax.set_title("3D t-SNE Visualization")
# ax.set_xlabel("Component 1")
# ax.set_ylabel("Component 2")
# ax.set_zlabel("Component 3")
# ax.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("IoV-tsne-3d", dpi=300)

#####################################################################################
# Create a 3D scatter plot using Plotly Express
fig = px.scatter_3d(
    tsne_df, 
    x="Component 1", 
    y="Component 2", 
    z="Component 3", 
    color="Label",             # Use the actual label names for coloring
    hover_data=["Label"],      # Optional: Adds a hover tooltip showing the Label
    title="Interactive 3D t-SNE Visualization"
)
# Adjust the marker size if needed
fig.update_traces(marker=dict(size=0.8))

# Save to an HTML file
fig.write_html("IoV-tsne-3d.html")