import torch

a = torch.tensor([2., 3.], requires_grad=True)
# 调用梯度请求
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a ** 3 - b ** 2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
# Q聚合成一个标量并隐式地向后调用

print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)
# 梯度沉淀在a.grad和b.grad
