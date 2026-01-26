import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

# 状態 [x, v] を入力、各 action の尤度 [l, n, r] を出力とするモデル
class QModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 必要なパラメタを定義 (例: self.l1 = nn.Linear(2, 10))
        self.l1 = nn.Linear(2, 80)
        self.l2 = nn.Linear(80, 80)
        self.l3 = nn.Linear(80, 80)
        self.l4 = nn.Linear(80, 3)

    def forward(self, x):
        # 順方向計算を書く (例: x = F.relu(self.l1(x)))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


model = QModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

xmax = -1

# 訓練データ (x: 入力, t: 教師)
xs = np.array([], dtype=np.float32)
ts = np.array([], dtype=np.float32)

BATCH_SIZE = 200

obs, info = env.reset(seed=42)
for i in range(1000000):
    x, v = obs
    xmax = max(xmax, x)
    if i % 10 == 0 or x > 0.3:
        print(i, f'record: {xmax}')
        env.render()

    # 方策
    model.eval()  # 推論モード
    with torch.inference_mode():
        y = model(torch.tensor(obs))
        probs = F.softmax(y, dim=0).detach().numpy()
        act = np.random.choice(3, p=probs)

        obs1, rew, term, trunc, info = env.step(act)

        # 訓練データを一つ作って積む
        t = y.detach().numpy()
        y1 = model(torch.tensor(obs1)).detach().numpy()
        t[act] = rew + max(y1)
        xs = np.concatenate((xs.reshape(-1, 2), obs.reshape(-1, 2)))
        ts = np.concatenate((ts.reshape(-1, 3), t.reshape(-1, 3)))

    obs = obs1

    if xs.shape[0] >= BATCH_SIZE:
        print('training...')
        model.train()
        for _ in range(500):
            opt.zero_grad()
            y = model(torch.tensor(xs))
            loss = loss_fn(y, torch.tensor(ts))
            loss.backward()
            opt.step()
        xs = np.array([], dtype=np.float32)
        ts = np.array([], dtype=np.float32)
    if term or trunc:
        obs, info = env.reset()

env.close()
