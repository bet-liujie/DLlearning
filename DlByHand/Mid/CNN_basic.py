import torch

in_channels, out_channels = 5, 10  # 定义输入输出的channel(通道)数
width, height = 100, 100  # 输入图像的长宽
kernel_size = 3  # 卷积和的大小,此处为3×3
batch_size = 1  # 最小批量

input = torch.randn(batch_size,
                    in_channels,
                    width, height)  # 输入四维的数据B×C×w×h

conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)  # 卷积层的建立，Conv2d函数，Conv2d(输入channel数，输出channel数，卷积和大小)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)  # 卷积层权重的形状

# padding操作,当你想输入输出shape一样时(或者设置输出shape)，在input进行padding操作(按照需求在外层填充0)
input = [3, 4, 5, 6, 7,
         2, 4, 6, 8, 8,
         2, 3, 1, 6, 9,
         6, 7, 5, 3, 0,
         3, 2, 1, 4, 9]
input = torch.Tensor(input).view(1, 1, 5, 5)  # view(B,C,W,H)
conv_layer = torch.nn.Conv2d(1, 1,
                             kernel_size=3,
                             padding=1,
                             bias=False)  # 输入输出channel都为1，卷积和为3，input周围一圈加满1，无偏置量

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)  # 卷积和的权重值设置,view(Out_channel,Input_channel,W,H)
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)

# stride(步长),卷积时方形移动步数设置，默认为1(遍寻)
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)

output = conv_layer(input)
print(output)  # 5×5的经过`卷`积和为3，stride为2的卷积后，变为2×2

# Max Pooling Layer(下采样的一种方法,默认stride=2，与通道数量无关,且无权重)

input = [1, 3, 5, 7,
         2, 4, 6, 8,
         7, 7, 3, 2,
         9, 7, 7, 1]

input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)
