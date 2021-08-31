import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 1个输入图像通道，6个输出通道，5x5平方卷积
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 一个仿射操作：y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 5*5来自图片尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # (2,2)窗口的最大池化
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        # 如果尺寸是方形，你可以指定一个特定的数
        x = torch.flatten(x, 1)
        # 展平除批次维度的其他维度
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
params = list(net.parameters())

print(len(params))
print(params[0].size())  # conv1(*卷积层1)的权重

input(torch.randn(1, 1, 32, 32))  # 输入一个随机的32×32
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))  # 使用随机梯度将所有的参数和反向传播的梯度缓冲区归零
