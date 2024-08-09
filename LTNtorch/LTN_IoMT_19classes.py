import torch
import pandas as pd
import ltn
import custom_fuzzy_ops
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 加载数据集
processed_train_file = '../CIC_IoMT/19classes/reduced_train_data.csv'
processed_test_file = '../CIC_IoMT/19classes/reduced_test_data.csv'

train_data = pd.read_csv(processed_train_file)
test_data = pd.read_csv(processed_test_file)