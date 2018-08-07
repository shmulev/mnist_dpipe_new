from os.path import join as jp
import gzip

import numpy as np
import torch.nn as nn
from torch.nn import functional

from dpipe.dataset import Dataset


def load_mnist(filename, offset):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


class MNIST(Dataset):
    def __init__(self, folder):
        self.xs = load_mnist(jp(folder, 'train-images-idx3-ubyte.gz'), 16).reshape(-1, 1, 28, 28).astype('float32')
        self.ys = load_mnist(jp(folder, 'train-labels-idx1-ubyte.gz'), 8).astype('long')
        self.ids = tuple(range(len(self.xs)))
        self.n_chans_image = 1

    def load_image(self, identifier: str):
        return self.xs[int(identifier)]

    def load_label(self, identifier: str):
        return self.ys[int(identifier)]


# model copied from pytorch examples:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)
