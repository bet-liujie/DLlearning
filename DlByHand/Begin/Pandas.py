import pandas as pd
import os

data_file = os.path.join('..', 'data', 'house_tiny.csv')  # 调用数据文件
data = pd.read_csv(data_file)  # 读取
print(data)
