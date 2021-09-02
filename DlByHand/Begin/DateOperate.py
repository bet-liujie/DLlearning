import torch  # 导入包

# 张量的基本创建，对应数学上的向量(单维度，ep:[1,2,3])或者矩阵(多维度)
x = torch.arange(12)  # 默认是行向量
print(x.shape)  # shape属性一般指行的元素数
