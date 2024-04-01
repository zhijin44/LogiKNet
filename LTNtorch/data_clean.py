import pandas as pd

# Replace these with the actual paths to your files
iris_training_path = 'datasets/iris_training_withzero.csv'

# Load the dataset
iris_data = pd.read_csv(iris_training_path)

# Replace 'species' values: changing 2 to 0
iris_data['species'] = iris_data['species'].replace(2, 0)

# Save the modified dataset back to csv
iris_data.to_csv(iris_training_path, index=False)

