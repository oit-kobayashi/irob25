import torch
import random

a = torch.tensor([0.1], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
opt = torch.optim.SGD([a, b], lr=0.00001)

for _ in range(10000):
    x = random.random() * 2.5 - 1.5
    t = (3 * x + 1)**3
    y = (a * x + b)**3
    loss = (t - y)**2
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a, b)
