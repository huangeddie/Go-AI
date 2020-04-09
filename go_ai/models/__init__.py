import os

import numpy as np
import torch
from mpi4py import MPI
from torch import nn as nn

from go_ai import data


class RLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_children = False
        self.assist = False
        self.nblocks = 1
        self.channels = 32

    def pt_critic(self, states):
        raise Exception("Not Implemented")

    def pt_actor(self, *args):
        raise Exception("Not Implemented")

    def pt_actor_critic(self, *args):
        raise Exception("Not Implemented")

    def create_numpy(self, mode):
        def np_func(states):
            return self._numpy(states, mode)

        return np_func

    def _numpy(self, states, mode, children=None):
        """
        :param states: Numpy batch of states
        :return:
        """
        invalid_values = data.batch_invalid_values(states)
        dtype = next(self.parameters()).type()

        # Execute on PyTorch
        pi_logits, val_logits = None, None
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)

            # Determine which pytorch function to call
            if mode == 'critic':
                val_logits = self.pt_critic(tensor_states)
            elif mode == 'actor' or mode == 'actor_critic':
                args = [tensor_states]
                # Compute children if needed
                if children is None and (self.requires_children or self.assist):
                    children = data.batch_padded_children(states)

                # Add it to args if requires children
                if self.requires_children:
                    tensor_ns = torch.tensor(children).type(dtype)
                    args.append(tensor_ns)

                if mode == 'actor':
                    pi_logits = self.pt_actor(*args)
                else:
                    assert mode == 'actor_critic'
                    pi_logits, val_logits = self.pt_actor_critic(*args)
            else:
                raise Exception(f"Unknown mode: {mode}")

        # Process PyTorch results
        if pi_logits is not None:
            pi_logits = pi_logits.detach().cpu().numpy()
            pi_logits += invalid_values
            if self.assist:
                # Set obvious moves
                # A child loss means we win and vice versa
                pi_logits += -100 * data.batch_win_children(children)

        if val_logits is not None:
            val_logits = val_logits.detach().cpu().numpy()
            if self.assist:
                # Set obvious values
                for i, state in enumerate(states):
                    if data.GoGame.get_game_ended(state):
                        val_logits[i] = 100 * data.GoGame.get_winning(state)

        # Return
        if pi_logits is None:
            return val_logits
        elif val_logits is None:
            return pi_logits
        else:
            return pi_logits, val_logits

    def train_step(self, optimizer, states, actions, reward, children, terminal, wins, pi):
        raise Exception("Not Implemented")

    def optimize(self, comm: MPI.Intracomm, batched_data, optimizer):
        raw_metrics = []
        self.train()
        for states, actions, reward, children, terminal, wins, pi in batched_data:
            metrics = self.train_step(optimizer, states, actions, reward, children, terminal, wins, pi)
            raw_metrics.append(metrics)

        # Sync Parameters
        average_model(comm, self)

        # Sync Metrics
        world_size = comm.Get_size()
        mean_metrics = np.mean(raw_metrics, axis=0)
        m = comm.allreduce(mean_metrics, op=MPI.SUM) / world_size

        metrics = ModelMetrics(m[0], m[1], m[2], m[3])

        # Return metrics
        return metrics


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


class ModelMetrics:
    def __init__(self, cl, ca, al, aa):
        self.crit_loss = cl
        self.crit_acc = ca

        self.act_loss = al
        self.act_acc = aa

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
