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
print(f"x + y ={x + y}")
print(f"x - y ={x - y}")
print(f"x * y ={x * y}")
print(f"x ** y ={x ** y}")
print(f"x / y ={x / y}\n")

# 新建一个形状为(3,4)数据类型为float32的从0到11的张量
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# cat函数连结两个张量，dim决定连结方式，0为按行连接，列不变，1则为按列连接，行不变,dim默认值为0
print(torch.cat((X, Y)))
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)

print(X.sum())  # sum对张量中所有元素求和，并产生一个只有一个元素的张量

# 广播机制
a = torch.arange(3).reshape((3, 1))  # 3×1矩阵
b = torch.arange(2).reshape((1, 2))  # 1×2矩阵
# shape不匹配(但一个矩阵的行要等于另一个矩阵的列)，相加后结果为3×2矩阵。
# 计算方式为每个单独矩阵复制增加到结果矩阵，然后相加。
print(f"a + b ={a + b}")
print(f"a * b ={a * b}\n")

# 索引与切片

# 第⼀个元素的索引是 0；可以指定范围以包含第⼀个元素和最后⼀个之前的元素。
# 我们可以通过使⽤负索引根据元素到列表尾部的相对位置访问元素。
a = torch.arange(8).reshape((4, -1))
print(a)
# [-1] 选择最后⼀个元素，⽤ [1:3] 选择第⼆个和第三个元素.
print(a[-1])
print(a[1:3])
# 通过指定索引来将元素写⼊矩阵。
a[1, 1] = 9
print(a)
a[0:2, :] = 8  # [0:2, :] 访问第1⾏和第2⾏，其中“:”代表沿轴 1（列）的所有元素
print(a)
a[:, 0:2] = 9
print(a)  # 类似

# 节省内存
# 运行一些操作会导致为新结果分配内存，在程序中可能会有大量的参数在一秒内大量改变。
# 我们希望不必要的分配内存，且希望能原地执行。
Z = torch.zeros_like(Y)
before = id(Y)
Y = X + Y  # 在计算X+Y后重新给Y分配了内存
print(id(Y) == before)
print('\nid(Z):', id(Z))
Z[:] = X + Y  # 使⽤切⽚表⽰法将操作的结果分配给先前分配的数组,ep:Y[:] = <expression>
print('id(Z):\n', id(Z))
# 如果在后续计算中没有重复使⽤ X，我们也可以使⽤ X[:] = X + Y 或 X += Y 来减少操作的内存开销
before = id(X)
X += Y
print(id(X) == before)

# 对象转换

A = x.numpy()
B = torch.tensor(A)
print(f"type(A)=,{type(A)}")
print(f"type(B)=,{type(B)}")  # 反之不再举例子
