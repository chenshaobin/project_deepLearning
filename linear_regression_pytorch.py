import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

torch.manual_seed(1)
# print(torch.rand(2))
print('torch version:', torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

# 初始化参数
num_inputs = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_example, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,   # 打乱数据
    num_workers=0
)

for X, y in data_iter:
    print('Just print one batch DataSet:')
    print('X:', X, '\n', 'y:', y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print('net 网络结构：', net)
# different solution to creat model:
"""
net_1 = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print('net_1 网络结构：', net_1)
net_2 = nn.Sequential()
net_2.add_module('linear', nn.Linear(num_inputs, 1))
print('net_2 网络结构：', net_2)
from collections import OrderedDict
net_3 = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
]))
print('net_3 网络结构：', net_3)
print('net_3 网络结构：', net_3[0])
"""

for param in net.parameters():
    print(param)
# 初始化模型参数
init.normal_(net.linear.weight, mean=0.0, std=0.01)
init.constant_(net.linear.bias, val=0.0)
print('初始化后的参数：')
for param in net.parameters():
    print(param)

loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
print('优化函数参数:', optimizer)
num_epoch = 10
for epoch in range(1, num_epoch + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l))

dense = net.linear
# print(dense)
print('true w:', true_w, '\n', 'train w', dense.weight.data)
print('true b:', true_b, '\n', 'train b', dense.bias.data)