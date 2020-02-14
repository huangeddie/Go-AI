import os

import gym
import torch
from mpi4py import MPI
from torch import nn as nn

from go_ai import data

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.main = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.main(x)
        out += identity
        out = torch.relu_(out)
        return out


def pytorch_ac_to_numpy(model):
    def ac_func(states):
        """
        :param states: Numpy batch of states
        :return:
        """
        invalid_values = data.batch_invalid_values(states)
        dtype = next(model.parameters()).type()
        model.eval()
        with torch.no_grad():
            tensor_states = torch.from_numpy(states).type(dtype)
            pi, state_vals = model(tensor_states)
            pi = pi.detach().cpu().numpy()
            pi += invalid_values
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return pi, vals

    return ac_func


def pytorch_val_to_numpy(model):
    """
    Automatically turns terminal states into 1, 0, -1 based on win status
    :param model: The model to convert
    :param val: True if value function, False if policy function
    :return: The numpy equivalent of the pytorch value model
    """

    def func(states):
        """
        :param states: Numpy batch of states
        :return:
        """
        dtype = next(model.parameters()).type()
        model.eval()
        with torch.no_grad():
            tensor_states = torch.from_numpy(states).type(dtype)
            state_vals = model(tensor_states)
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return vals

    return func


class ModelMetrics:
    def __init__(self):
        self.crit_acc = 0
        self.crit_loss = 0
        self.act_acc = 0
        self.act_loss = 0

    def __str__(self):
        critic = f'C[{self.crit_acc * 100 :.1f}% {self.crit_loss:.3f}L]'
        actor = f'A[{self.act_acc * 100 :.1f}% {self.act_loss:.3f}L]'
        return f'{critic} {actor}'

    def __repr__(self):
        return self.__str__()


def average_model(comm, model):
    world_size = comm.Get_size()
    for params in model.parameters():
        params.data = comm.allreduce(params.data, op=MPI.SUM) / world_size


def get_modelpath(args, savetype):
    if savetype == 'checkpoint':
        dir = args.checkdir
    elif savetype == 'baseline':
        dir = 'bin/baselines/'
    else:
        raise Exception(f"Unknown location type: {savetype}")
    path = os.path.join(dir, f'{args.model}{args.size}.pt')

    return path