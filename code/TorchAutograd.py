import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data)
loss = (prediction - labels).sum()
loss.backward()
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()
print(optim)
