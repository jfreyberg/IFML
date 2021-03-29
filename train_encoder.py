import numpy as np
from torch.autograd import Variable
from torch import nn
import torch
import time
import pickle
import statistics

from pprint import pprint
import copy
import settings as s
import events as e

import matplotlib.pyplot as plt

import os

N = 4
EPOCHS = 20

PLOT_SAMPLES = 6

fig, axs = plt.subplots(2, 7)
fig = plt.figure(figsize=((.75*6, .75*(1+1+PLOT_SAMPLES*2))))
axs = []
axs.append(plt.subplot2grid((1+PLOT_SAMPLES*2, 6), (0, 0), colspan=6))
axs[0].set_yscale('log')

for j in range(2*PLOT_SAMPLES):
    row = []
    for i in range(6):
        row.append(plt.subplot2grid((1+PLOT_SAMPLES*2, 6), (j+1, i)))
    axs.append(row)


def debug_plot(loss, X, Y, Z):
    axs[0].plot(loss)
    for i in range(6):
        for j in range(PLOT_SAMPLES):
            axs[1+j*2][i].imshow(X[j][i], vmin=0, vmax=1)
            axs[1+j*2][i].set_axis_off()
            axs[2+j*2][i].imshow(Y[j][i], vmin=0, vmax=1)
            axs[2+j*2][i].set_axis_off()

    plt.savefig("data.png")
    for ax2 in axs:
        if type(ax2) == list:
            for ax in ax2:
                ax.clear()
        else:
            ax2.clear()


def custom_loss(x, y):
    x_enemy_pos          = x[:, 0, None]
    x_enemy_bombs        = x[:, 1, None]
    x_coins              = x[:, 2, None]
    x_can_destroy        = x[:, 3, None]
    x_full_explosion_map = x[:, 4, None]
    x_can_enter          = x[:, 5, None]

    y_enemy_pos          = y[:, 0, None]
    y_enemy_bombs        = y[:, 1, None]
    y_coins              = y[:, 2, None]
    y_can_destroy        = y[:, 3, None]
    y_full_explosion_map = y[:, 4, None]
    y_can_enter          = y[:, 5, None]

    enemy_pos_loss = nn.L1Loss()(x_enemy_pos, y_enemy_pos)
    enemy_bombs_loss = nn.L1Loss()(x_enemy_bombs, y_enemy_bombs)
    coins_loss = nn.L1Loss()(x_coins, y_coins)
    can_destroy_loss = nn.MSELoss()(x_can_destroy, y_can_destroy)
    full_explosion_map_loss = nn.L1Loss()(x_full_explosion_map, y_full_explosion_map)
    can_enter_loss = nn.MSELoss()(x_can_enter, y_can_enter)

    sparse_loss = (enemy_pos_loss + enemy_bombs_loss + coins_loss + full_explosion_map_loss)
    dense_loss = (can_enter_loss + can_destroy_loss)

    sparse_factor = 1/(torch.sum(x_enemy_pos) + 
                       torch.sum(x_enemy_bombs) +
                       torch.sum(x_coins) +
                       torch.sum(x_full_explosion_map))
    sparse_factor *= 10

    dense_factor = 1/(torch.sum(x_can_enter) + 
            torch.sum(x_can_destroy))

    if sparse_factor*sparse_loss < dense_factor*dense_loss:
        sparse_factor *= 10
    else:
        dense_factor *= 10


    return 10000*(sparse_factor * sparse_loss + dense_factor * dense_loss)

class agent(nn.Module):
    def __init__(self, window_size=N, hidden=128, embed=32, learning_rate=5e-3):
        super(agent, self).__init__()

        self.window_size = window_size
        self.hidden = hidden
        self.embed = embed
        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(
            nn.Linear((self.window_size*2+1)**2*6, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, embed)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, (self.window_size*2+1)**2*6),
            nn.Hardsigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def forward(self, x):
        #x = torch.from_numpy(x).type(torch.FloatTensor)
        x = x.reshape((-1, (N*2+1)**2*6))

        z = self.encoder(x)
        y = self.decoder(z)

        y = y.reshape((-1, 6, self.window_size*2+1, self.window_size*2+1))

        #x[0] = x[0] * 25
        #x[1] = x[1] * 6
        #x[4] = x[4] * 5
        return y

    def encode(self, x):
        #x = torch.from_numpy(x).type(torch.FloatTensor)
        x = x.reshape((-1, (N*2+1)**2*6))

        z = self.encoder(x)

        #z = z.reshape((-1, self.window_size*2+1, self.window_size*2+1))

        #x[0] = x[0] * 25
        #x[1] = x[1] * 6
        #x[4] = x[4] * 5
        return z


data = []

print("loading...")

try:
    data = pickle.load(open('replays.pkl', 'rb'))
except:
    for i, filename in enumerate(os.listdir('own_replays')):
        res = np.load(f'own_replays/{filename}')

        # normalize
        res[0] = res[0] / 25
        res[1] = res[1] / 6
        res[4] = res[4] / 5

        data.append(torch.Tensor(res))

    pickle.dump(data, open('replays.pkl', 'wb'))

print(f"loaded {len(data)} elements")

dataLoader = torch.utils.data.DataLoader(dataset=data,
                                         batch_size=500,
                                         shuffle=True)

model = agent(N, 81*3, 81)

losses = []

ts = time.time()

for epoch in range(EPOCHS):
    for data in dataLoader:
        # Predict
        pred = model(data)

        # Loss
        #loss = self.criterion(output, arena)
        loss = custom_loss(pred, data)

        # Back propagation
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        losses.append(loss.item())

        if ts + 5 < time.time():
            with torch.no_grad():
                print('epoch [{}/{}], loss:{:.4f}, stddev: {:.4f}'.format(epoch + 1, EPOCHS, loss.data, statistics.stdev(losses)))
                losses = losses[-5000:]
                encoded = model.encode(data)
                debug_plot(losses, data.detach(), pred.detach(), encoded.detach())
                ts = time.time()

    torch.save(model, 'trained.torch')
    torch.save(model.state_dict(), 'trained_dict.torch')
