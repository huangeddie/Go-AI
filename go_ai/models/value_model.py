import gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from go_ai import data

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ValueNet(nn.Module):
    def __init__(self, board_size):
        super(ValueNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(GoVars.NUM_CHNLS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU(),
        )

        self.fcs = nn.Sequential(
            nn.Linear(board_size ** 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.optim = optim.Adam(self.parameters(), 1e-2)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def optimize(self, replay_data, batch_size):
        N = len(replay_data[0])
        for component in replay_data:
            assert len(component) == N

        batched_data = [np.array_split(component, N // batch_size) for component in replay_data]

        self.train()
        running_loss = 0
        running_acc = 0
        pbar = tqdm(zip(*batched_data), desc="Optimizing", position=0)
        for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
            # Augment
            states = data.batch_random_symmetries(states)

            states = torch.from_numpy(states).type(torch.FloatTensor)
            wins = torch.from_numpy(wins[:, np.newaxis]).type(torch.FloatTensor)

            self.optim.zero_grad()
            vals = self(states)
            pred_wins = (torch.sigmoid(vals) > 0.5).type(torch.FloatTensor)
            loss = self.criterion(vals, wins)
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            running_acc += torch.mean((pred_wins == wins).type(torch.FloatTensor)).item()

            pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100* running_acc / i, running_loss / i))
