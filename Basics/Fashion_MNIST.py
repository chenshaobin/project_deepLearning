import torch
import torchvision
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
# print(sys.path)
import Utils as util
print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True,download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
print(type(mnist_train))
print('mnist train length:', len(mnist_train), '\n', 'mnist test length:', len(mnist_test))

feature, label = mnist_train[0]
print('Just Print one example:')
print('feature shape:', feature.shape, '\n', 'feature type:', type(feature))
print('label:', label)
# mnist_PIL = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST', train=True,download=True)
# PIL_feature, label = mnist_PIL[0]
# print(PIL_feature)

print('Test image show Function:')
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

# util.show_Fashion_MNIST(X, util.get_Fashion_MNIST_labels(y))      # show image
# print(sys.platform)
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
