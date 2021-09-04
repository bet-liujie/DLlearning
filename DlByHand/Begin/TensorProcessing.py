import torch

A = torch.arange(20, dtype=torch.float32).reshape((4, -1))
print(f"A={A}\n", f"A.T={A.T}\n")  # A的转置矩阵

B = A.clone()  # 通过分配新内存，将A的⼀个副本分配给B
print(A == B)
a = 2
B[:] = a + B
print(f"B={B}\t", f"B.shape = {B.shape}")

# 降维
print(A.sum)  # 直接降维为一个标量
A_sum_axis0 = A.sum(axis=0)  # 指定维度降维，axis=0时，维度0消失(也可以说变为1)
print(f"A_sum_axis0=\n{A_sum_axis0}\n", f"A_sum_axis0.shape={A_sum_axis0.shape}")
print(f"A_sum_axis1=\n{A.sum(axis=1)}\n", f"A_sum_axis1.shape={A.sum(axis=1).shape}")
