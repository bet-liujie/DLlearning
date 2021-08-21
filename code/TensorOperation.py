import torch
import numpy as np

tensor = torch.ones(4, 4)  # 类似numpy的切片操作
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

# 张量的乘法（元素乘法），两种方式
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor * tensor \n{tensor * tensor}")

# 张量的矩阵乘法，两种方式
print(f"tensor.matmul(tensor.T) \n{tensor.matmul(tensor.T)}\n")
print(f"tensor @ tensor.T \n{tensor @ tensor.T}\n")

# 就地操作，带_后缀的操作就是就地操作
print(tensor)
tensor.add_(2)
print(tensor)
# 可以节省一些内存，但计数导数可能出问题，可能会丢失历史记录，因此不鼓励使用

# NumPy桥接，cpu和numpy数组上的张量共享底层内存位置

# 张量到数组
t = torch.ones(3)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# 数组到张量
n = np.ones(4)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
