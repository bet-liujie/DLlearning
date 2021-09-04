import torch

A = torch.arange(12, dtype=torch.float32).reshape((-1, 4))
print(f"A={A}\n", f"A.T={A.T}\n")  # A的转置矩阵

B = A.clone()  # 通过分配新内存，将A的⼀个副本分配给B
print(A == B)
a = 2
B[:] = a + B
print(f"B={B}\t", f"B.shape = {B.shape}")

# 降维
print(A.sum)  # 直接降维为一个标量(求和)
A_sum_axis0 = A.sum(axis=0)  # 指定维度降维，axis=0时，维度0消失(也可以说变为1),该维度进行求和
print(f"A_sum_axis0=\n{A_sum_axis0}\n", f"A_sum_axis0.shape={A_sum_axis0.shape}")
print(f"A_sum_axis1=\n{A.sum(axis=1)}\n", f"A_sum_axis1.shape={A.sum(axis=1).shape}")
print(A.mean)  # 求平均，也可降维，同求和

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(f"sum_A=\n{sum_A}")
print(f"A / sum_A=\n{A / sum_A}")  # 利用广播机制保留维度
print(f"A.cumsum=\n{A.cumsum(axis=0)}\n")  # 利用cumsum保留维度求和

# 几种线性代数运算
b = torch.ones(4, dtype=torch.float32)
print(f"b={b}\n", f"b*b={torch.dot(b, b)}\n")  # 点积
print(f"Ab=\n{torch.mv(A, b)}")  # 矩阵向量积
B = torch.ones(4, 2)
print(f"AB=\n{torch.mm(A, B)}")  # 矩阵乘积

# 范数
u = torch.tensor([3.0, -4.0])
print(f"u的范数:{torch.norm(u)}")  # 深度学习中常使用L2范数，本例为L2范数
print(f"u的L1范数:{torch.abs(u).sum()}")  # u为2维度，其L1范数的计算为其元素绝对值求和
print(f"B的弗罗贝尼乌斯范数:{torch.norm(B)}")  # 弗罗贝尼乌斯范数:一种矩阵范数，算法与向量范数类似
