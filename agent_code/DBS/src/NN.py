import torch.nn.functional as F
from torch import nn
import torch
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, input_dim, extra_dim, action_dim, learnin_rate, network_arch=None, is_training=False):
        super().__init__()
        c, h, w = input_dim
        self.input_dim = input_dim
        self.extra_dim = extra_dim
        self.network_arch = network_arch

        self.flatten = nn.Flatten()

        if network_arch is None:
            drop_prob = 0
            if is_training:
                drop_prob = 0.15

            self.lin1 = nn.Linear(c*h*w + extra_dim, c*h*w//2)
            self.lin2 = nn.Linear(c*h*w//2, c*h*w//3)
            self.lin3 = nn.Linear(c*h*w//3, c*h*w//2)
            self.lin4 = nn.Linear(c*h*w//2, h*w)
            self.lin5 = nn.Linear(h*w, action_dim)
        else:

            network_arch = {
                "layers_x1": [],
                "layers_x2": [],
                "layers_x": [
                    {"out": 2},
                    {"out": 3},
                    {"out": 2}
                ],
                "dropout": 0.15,
                "layers_end": [
                    {"out": 1}
                ]
            }

            drop_prob = 0
            if is_training:
                drop_prob = network_arch['dropout']

            # self.conv1 = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1)
            self.lins_x1 = nn.ModuleList()
            self.lins_x2 = nn.ModuleList()
            self.lins_x = nn.ModuleList()
            self.lins_end = nn.ModuleList()

            in_size = c*h*w+extra_dim

            for layer in network_arch['layers_x1']:
                out_size = int(c*h*w / layer['out'])
                self.lins_x1.append(nn.Linear(in_size, out_size))
                in_size = out_size

            for layer in network_arch['layers_x2']:
                out_size = int(c*h*w / layer['out'])
                self.lins_x2.append(nn.Linear(in_size, out_size))
                in_size = out_size

            for layer in network_arch['layers_x']:
                out_size = int(c*h*w / layer['out'])
                self.lins_x.append(nn.Linear(in_size, out_size))
                in_size = out_size

            for layer in network_arch['layers_end']:
                out_size = int(c*h*w / layer['out'])
                self.lins_end.append(nn.Linear(in_size, out_size))
                in_size = out_size

            self.lins_end.append(nn.Linear(in_size, action_dim))

        self.dropout = nn.Dropout(p=drop_prob)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learnin_rate)
        self.loss = nn.MSELoss()

    def forward(self, x1, x2):

        if self.network_arch is None:
            x1 = self.flatten(x1)
            if self.extra_dim > 0:
                x = torch.cat((x1, x2), 1)
            else:
                x = x1

            x = F.relu(self.lin1(x))
            x = F.relu(self.lin2(x))
            x = F.relu(self.lin3(x))
            x = self.dropout(x)
            x = F.relu(self.lin4(x))
            x = self.lin5(x)
        else:
            x1 = self.flatten(x1)

            for layer in self.lins_x1:
                x1 = F.relu(layer(x1))

            for layer in self.lins_x2:
                x2 = F.relu(layer(x2))

            if self.extra_dim > 0:
                x = torch.cat((x1, x2), 1)
            else:
                x = x1

            for layer in self.lins_x:
                x = F.relu(layer(x))

            for layer in self.lins_end[:-1]:
                x = F.relu(layer(x))

            x = self.lins_end[-1](x)

        return x
