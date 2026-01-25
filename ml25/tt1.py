import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

N = 5
x = np.random.random((N, 2)).astype(np.float32) * 6 - 3

_x = x * x * np.array([1/4, 1])
tf = _x[:, 0] + _x[:, 1] < 1
y = np.zeros((N, 2))
y[tf == True] = [0, 1]
y[tf != True] = [1, 0]

xt = torch.from_numpy(x)
yt = torch.from_numpy(y)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 2)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(x, dim=1)
        return x
