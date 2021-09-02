import torch  # 导入包

# 张量的基本创建，对应数学上的向量(单维度，ep:[1,2,3])或者矩阵(多维度)
x = torch.arange(12)  # 默认是行向量
print(x.shape)  # shape属性一般指沿行的元素数(ep:[4,2,4,5].shape==4)
print(x.numel())  # x.numel()为张量的size，即其对应矩阵或向量的中的元素数

X = x.reshape(3, 4)  # reshape函数改变张量的形状(内容不变)，该例种把(12,)的行向量转变为(3,4)的矩阵
# X = x.reshape(-1, 4) -1调用让它自己计算其应该对应的维度
# X = x.reshape(3, -1)
print(X)

# 张量的初始化
m = torch.zeros((2, 3, 4))  # 全0初始化
print(m)
m = torch.ones((2, 3, 4))  # 全1初始化
print(m)

# 张量的数据操作
A = torch.randn(3, 4)  # 随机的数据创建，数据来自均值为0，标准差为1的标准正态分布
print(A)
A = torch.tensor([[1, 2, 3, 4], [3, 4, 9, 8]])  # 直接初始其中的值
print(A)

# 按元素运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(f"x + y={x + y}")
print(f"x - y={x - y}")
print(f"x * y={x * y}")
print(f"x ** y={x ** y}")
print(f"x / y={x / y}")

# 新建一个形状为(3,4)数据类型为float32的从1到12的张量
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# cat函数连结两个张量，dim决定连结方式，0为按行连接，列不变，1则为按列连接，行不变,dim默认值为0
print(torch.cat((X, Y)))
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)

print(X.sum())  # sum对张量中所有元素求和，并产生一个只有一个元素的张量

