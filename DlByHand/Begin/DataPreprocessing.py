import os

# 在py文件路径下配置一个文件夹‘data’，并创建一个名叫house_tiny的csv文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名:房间数量,巷子类型，价格
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,156486\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
