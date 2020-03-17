import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

from go_ai import data
from go_ai.models import BasicBlock, average_model, ModelMetrics

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ValueNet(nn.Module):
    """
    ResNet
    """

    def __init__(self, size, num_blocks=4, channels=32):
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
            nn.Conv2d(channels, 1, 1),
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

    def val_numpy(self, states):
        dtype = next(self.parameters()).type()
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)
            state_vals = self(tensor_states)
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return vals

    def optimize(self, comm: MPI.Intracomm, batched_data, optimizer):
        world_size = comm.Get_size()
        self.train()
        dtype = next(self.parameters()).type()
        running_loss = 0
        running_acc = 0
        for states, _, _, _, _, wins, _ in batched_data:
            # Augment
            states = data.batch_random_symmetries(states)

            states = torch.tensor(states).type(dtype)
            wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

            optimizer.zero_grad()
            logits = self(states)
            vals = torch.tanh(logits)
            pred_wins = torch.sign(vals)
            loss = self.criterion(vals, wins)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_acc += torch.mean((pred_wins == wins).type(wins.dtype)).item()

        # Sync Parameters
        average_model(comm, self)

        running_acc = comm.allreduce(running_acc, op=MPI.SUM) / world_size
        running_loss = comm.allreduce(running_loss, op=MPI.SUM) / world_size

        metrics = ModelMetrics()
        metrics.crit_acc = running_acc / len(batched_data)
        metrics.crit_loss = running_loss / len(batched_data)

        return metrics
