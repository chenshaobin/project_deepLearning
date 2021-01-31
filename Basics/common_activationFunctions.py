import torch
import numpy as np
import sys
sys.path.append("..")
import Utils as utils
print('torch version:', torch.__version__)

def xyplot(x_vals, y_vals, name):
    utils.set_figsize(figsize=(5, 2.5))
    utils.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    utils.plt.xlabel('x')
    utils.plt.ylabel(name + '(x)')

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# relu
"""
y = torch.relu(x)
xyplot(x, y, 'relu')
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
utils.plt.show()
"""
# sigmoid
"""
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
utils.plt.show()
"""
# tanh
y = x.tanh()
xyplot(x, y, 'sigmoid')
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
utils.plt.show()