import torch
from torch import nn
from torch.nn import init
import sys
import Utils as utils
sys.path.append("..")

print('torch version:', torch.__version__)

# 定义模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = nn.Sequential(
    utils.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

for param in net.parameters():
    init.normal_(param, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, None,  None, optimizer)
