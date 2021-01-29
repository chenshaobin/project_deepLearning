import torch
from time import time
from matplotlib import pyplot as plt
import numpy as np
import random


print('torch version:', torch.__version__)
"""
a = torch.ones(1000)
b = torch.ones(1000)
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print('#1: run time:', time() - start)
start = time()
d = a + b
print('#2: run time:', time() - start)
aa = torch.ones(3)
bb = 10
print(aa + bb)
"""

num_inputs = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_example, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# print(features[0], labels[0])
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # print('number of examples:', num_examples)
    indices = list(range(num_examples))
    random.shuffle(indices)     # 随机调整顺序
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
print('Print one patch data:')
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        # print(param.grad)
        param.data -= lr * param.grad / batch_size

lr = 0.01
num_epoch = 5
net = linreg
loss = square_loss
for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch: %d, loss: %f' % (epoch + 1, train_l.mean().item()))

print('true w:', true_w, '\n', 'train w:', w)
print('true b:', true_b, '\n', 'train b:', b)
