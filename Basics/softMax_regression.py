import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")
import Utils as utils
print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
# 获取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
# 实现softmax运算
def softmax(X):
    X_exp = X.exp()
    col_sum_exp = X_exp.sum(dim=1, keepdim=True)
    return X_exp / col_sum_exp

# X = torch.rand((2,5))
# X_prob = softmax(X)
# print('X_prop:', X_prob, '\n', 'sum X_prop:', X_prob.sum(dim=1))
# 定义模型

def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs), W) + b)

# 定义损失函数
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# # print(y.view(-1,1))

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 计算准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# print(accuracy(y_hat, y))
# print(utils.evaluate_accuary(test_iter, net))

# 训练模型
num_epochs, lr = 5, 0.03
utils.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W,b], lr)
# 预测
X, y = iter(test_iter).next()
true_labels = utils.get_Fashion_MNIST_labels(y.numpy())
prep_labels = utils.get_Fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + prep for true, prep in zip(true_labels, prep_labels)]
print(titles[0:9])
utils.show_Fashion_MNIST(X[0:9], titles[0:9])

