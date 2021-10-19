import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 准备数据集
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# CNN模型建立
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))  # 卷积层1，输入通道为1，输出通道为10，卷积和为5(5×5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))  # 卷积层2,输入通道为10，输出为10，卷积和为5
        self.pooling = torch.nn.MaxPool2d(2)  # 最大池化，stride为2
        self.fc = torch.nn.Linear(320, 10)  # 最后输出线性量,320为计算出的值(每个卷积层之间都有最大池化)

    def forward(self, x):
        batch_size = x.size(0)  # 算出x的维度
        x = F.relu(self.pooling(self.conv1(x)))  # 这里是先卷积后池化，顺序无所谓
        x = F.relu(self.pooling(self.conv2(x)))  # 第二次relu激活
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 启用gpu
model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 优化器


# training cycle forward, backward, update


def train(epoch):  # 训练
    running_loss = 0.0  # 实时损失
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 启用GPU
        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()  # 梯度返回
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
