import torch
import numpy as np

tensor = torch.ones(4, 4)  # 类似nump的切片操作
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    # 转移到gpu上运行（速度通常比cpu快），所有的张量操作都可以再gpu上运行
tensor[:, 2] = 5
print(tensor)
tensor[3, :] = 0
print(tensor)
# :的位置表示操作的是行还是列，放前面操作的是行，后面是列。
# 另外一个数字表示操作的行数-1（可能遵从数组的原则从0开始计数）

t1 = torch.cat([tensor, tensor, tensor], dim=0)  # dim表示连接维度，0为行增连接
t2 = torch.cat([tensor, tensor, tensor], dim=1)  # 1为列增加连接
t3 = torch.cat([tensor, tensor])  # dim默认为0
print(t1)
print(t2)
print(t3)
