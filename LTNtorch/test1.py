import pandas as pd

# 定义文件路径
training_file = 'datasets/iris_training_withzero.csv'
test_file = 'datasets/iris_test_withzero.csv'

# 处理训练数据集
train_data = pd.read_csv(training_file)
# 将第三列的值设置为0
train_data.iloc[:, 2] = 0
# 保存修改后的DataFrame回文件
train_data.to_csv(training_file, index=False)

# 处理测试数据集
test_data = pd.read_csv(test_file)
# 将第三列的值设置为0
test_data.iloc[:, 2] = 0
# 保存修改后的DataFrame回文件
test_data.to_csv(test_file, index=False)

print("第三列值已成功修改为0。")
