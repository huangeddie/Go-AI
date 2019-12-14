import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

from go_ai import data, measurements
from go_ai.models import BasicBlock

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ValueNet(nn.Module):
    """
    ResNet
    """

    def __init__(self, boardsize, num_blocks=4, channels=32):
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

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x


def optimize(comm: MPI.Intracomm, model: torch.nn.Module, batched_data, optimizer):
    world_size = comm.Get_size()
    model.train()
    dtype = next(model.parameters()).type()
    running_loss = 0
    running_acc = 0
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(batched_data, 1):
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(dtype)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(dtype)

        optimizer.zero_grad()
        logits = model(states)
        vals = torch.tanh(logits)
        pred_wins = torch.sign(vals)
        loss = model.criterion(vals, wins)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_acc += torch.mean((pred_wins == wins).type(wins.dtype)).item()

    # Sync Parameters
    for params in model.parameters():
        params.data = comm.allreduce(params.data, op=MPI.SUM) / world_size

    running_acc = comm.allreduce(running_acc, op=MPI.SUM) / world_size
    running_loss = comm.allreduce(running_loss, op=MPI.SUM) / world_size

    metrics = measurements.ModelMetrics()
    metrics.crit_acc = running_acc / len(batched_data)
    metrics.crit_loss = running_loss / len(batched_data)

    return metrics
