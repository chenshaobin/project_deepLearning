import torch
import numpy as np
import sys
import Utils as utils
sys.path.append("..")

print('torch version:', torch.__version__)
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化模型参数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1,b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#定义模型
def net(X):
    X = X.view(-1, num_inputs)
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = torch.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)
