import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

N = 5000
x = np.random.random((N, 2)).astype(np.float32) * 2 - 1

_x = x * x * np.array([5, 1.5])
# _x2 =  x * x * np.array([1.2, 6])
tf = _x[:, 0] + _x[:, 1] < 1
# tf = np.logical_or(_x[:, 0] + _x[:, 1] < 1,  _x2[:, 0] + _x2[:, 1] < 1)
y = np.zeros((N, 2))
y[tf == True] = [0, 1]
y[tf != True] = [1, 0]

xt = torch.from_numpy(x)
yt = torch.from_numpy(y)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 2)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x

# ↑ここまで #13 の授業

# tf で色を変えつつ訓練データ (正解) をプロット
sc0 = plt.scatter(x[tf, 0], x[tf, 1], s=1, c="blue")
sc1 = plt.scatter(x[tf!=True, 0], x[tf!=True, 1], s=1, c="red")
plt.axis('scaled')  # アスペクト比を 1:1 に
plt.ion()           # interactive mode というのに入るらしい
plt.waitforbuttonpress()  # キー待ち

model = MyModel()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()  # loss は関数 (を代入した変数)

EPOCHS = 10000
for i in range(EPOCHS):
    opt.zero_grad()       # 前回の勾配をクリア
    y_pred = model(xt)    # 順方向計算
    l = loss(yt, y_pred)  # 計算グラフの最後は損失計算
    l.backward()          # 勾配計算
    opt.step()            # パラメタ更新

    # ここからはビジュアライゼーションで本質ではない
    tf_pred = y_pred[:, 0] > 0.5
    if i % 10 == 0:
        # 予測結果をプロット
        print(i, l)
        sc0.set_offsets(xt[tf_pred, :])
        sc1.set_offsets(xt[tf_pred != True, :])
        plt.pause(0.01)
        
plt.waitforbuttonpress()
