import torch
from torch import nn
from torch.nn import init
import torchvision
import numpy as np
from collections import OrderedDict


import sys
sys.path.append("..")
import Utils as utils
print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)

# 获取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

#定义和初始化模型
num_inputs = 784
num_outputs = 10

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
print('init weight:', net.linear.weight, '\n', 'init bias:', net.linear.bias)
loss = nn.CrossEntropyLoss()    # 包含交叉熵损失和softmax操作
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 5
# (net, train_iter, test_iter, loss, num_epochs,batch_size, params=None, lr=None, optimizer=None)
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
