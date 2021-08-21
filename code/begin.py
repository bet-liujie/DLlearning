import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # 张量初始话，直接创建
print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)  # 张量可以用nump数组创建
print(x_np)
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n {x_ones}\n")  # 张量x_ones未改变其属性
x_rand = torch.rand_like(x_data, dtype=torch.float)  # 属性，数据全改变
print(f"Random Tensor: \n{x_rand} \n")

shape = (3, 4,)  # shape是张量维度的元组，决定了输出张量的维度
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor} \n")
print(f"Ones Tensor: \n{ones_tensor} \n")
print(f"Zeros Tensor: \n{zeros_tensor} \n")

tensor = torch.rand(3, 4)  # 张量的属性

print(f"Shape of tensor: {tensor.shape}")  # 张量的形状（行列数）
print(f"Datatype of tensor: {tensor.dtype}")  # 张量的数据类型 这里是float32
print(f"Device tensor is stored on :{tensor.device}")  # 张量的存储设备 这里是cpu
