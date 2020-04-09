import numpy as np
import torch
import torch.nn as nn

from go_ai import data
from go_ai.models import BasicBlock, RLNet


class ValueNet(RLNet):
    def __init__(self, size):
        super().__init__()

        # Convolutions
        convs = [
            nn.Conv2d(6, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        ]

        for i in range(self.nblocks):
            convs.append(BasicBlock(self.channels, self.channels))

        convs.extend([
            nn.Conv2d(self.channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ])

        self.convs = nn.Sequential(*convs)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.convs(x)

    def pt_critic(self, states):
        return self.forward(states)

    def train_step(self, optimizer, _, actions, rewards, children, terminal, wins, pi):
        dtype = next(self.parameters()).type()
        bsz = len(children)

        # Augment
        next_states = children[np.arange(bsz), actions]
        next_states = data.batch_random_symmetries(next_states)

        # To tensors
        next_states = torch.tensor(next_states).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Loss
        optimizer.zero_grad()
        logits = self(next_states)
        vals = -torch.tanh(logits)
        pred_wins = torch.sign(vals)
        loss = self.criterion(vals, wins)
        loss.backward()

        optimizer.step()

        # Accuracy
        acc = torch.mean((pred_wins == wins).type(wins.dtype)).item()

        return loss.item(), acc, 0, 0
