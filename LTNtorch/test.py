import os
import pandas as pd
print(os.getcwd())

df = pd.read_csv("datasets/crabs.dat", sep=" ", skipinitialspace=True)