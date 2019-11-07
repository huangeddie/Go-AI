import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from go_ai import data

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ValueNet(nn.Module):
    """
    ResNet
    """
    def __init__(self, boardsize, num_blocks=8, channels=32):
        super().__init__()

        # Convolutions
        convs = [
            nn.Conv2d(6, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]

        for i in range(num_blocks):
            convs.append(BasicBlock(channels, channels))

        convs.extend([
            nn.Conv2d(channels, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        ])

        self.convs = nn.Sequential(*convs)

        # Fully Connected
        fc_h = 4 * boardsize ** 2
        self.fcs = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x


def optimize(model: torch.nn.Module, batched_data, optimizer):

    model.train()
    dtype = next(model.parameters()).dtype
    running_loss = 0
    running_acc = 0
    batches = 0
    pbar = tqdm(batched_data, desc="Optimizing", leave=True)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(dtype)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(dtype)

        optimizer.zero_grad()
        vals = model(states)
        pred_wins = (torch.sigmoid(vals) > 0.5).type(vals.dtype)
        loss = model.criterion(vals, wins)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += torch.mean((pred_wins == wins).type(wins.dtype)).item()
        batches = i

        pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * running_acc / i, running_loss / i))

    return running_acc / batches, running_loss / batches
