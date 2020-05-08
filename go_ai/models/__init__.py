import os
import warnings

import numpy as np
import torch
from mpi4py import MPI
from torch import nn as nn
from torch.nn import functional as F

from go_ai import data


class RLNet(nn.Module):
    def __init__(self, in_c=6):
        super().__init__()
        self.requires_children = False
        self.assist = True
        self.layers = 2
        self.channels = 128

        # Convolutions
        convs = [
            nn.Conv2d(in_c, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        ]

        for i in range(self.layers):
            convs.append(BasicBlock(self.channels, self.channels))

        self.main = nn.Sequential(*convs)

    def pt_critic(self, states):
        raise Exception("Not Implemented")

    def pt_actor(self, *args):
        raise Exception("Not Implemented")

    def pt_actor_critic(self, *args):
        raise Exception("Not Implemented")

    def pt_game(self, states):
        raise Exception("Not Implemented")

    def create_numpy(self, mode):
        def np_func(states):
            return self._numpy(states, mode)

        return np_func

    def dtype(self):
        return next(self.parameters()).type()

    def _numpy(self, states, mode, children=None):
        """
        :param states: Numpy batch of states
        :return:
        """
        invalid_values = data.batch_invalid_values(states)
        dtype = self.dtype()

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

    def critic_step(self, imgs, wins):
        dtype = self.dtype()

        # Preprocess
        imgs = data.batch_random_symmetries(imgs)

        # To tensors
        imgs = torch.tensor(imgs).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Critic Loss
        val_logits = self.pt_critic(imgs)
        vals = torch.tanh(val_logits)

        critic_loss = F.mse_loss(vals, wins)

        # Predict wins
        pred_wins = torch.sign(vals)
        critic_acc = torch.mean((pred_wins == wins).type(dtype))
        return critic_loss, critic_acc.item()

    def reinforce_step(self, states, children, actions, wins):
        dtype = self.dtype()
        bsz = len(states)

        # To tensors
        # Do not augment with random symmetries. Otherwise it invalidates the actions
        states = torch.tensor(states).type(dtype)
        children = torch.tensor(children).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Value for baseline
        with torch.no_grad():
            val_logits = self.pt_critic(states)
            vals = torch.tanh(val_logits)

        # Forward pass
        pi_logits = self.pt_actor(states, children)

        # Log probability of taken actions
        logpi_logits = torch.log_softmax(pi_logits, dim=1)
        log_pis = logpi_logits[torch.arange(bsz), actions].reshape(-1, 1)

        # Policy Gradient
        assert vals.shape == wins.shape
        advantages = wins - vals
        assert log_pis.shape == advantages.shape
        expected_reward = log_pis * advantages
        actor_loss = -torch.mean(expected_reward)

        return actor_loss, 0

    def actor_step(self, states, pi):
        dtype = self.dtype()

        # Do not augment with random symmetries because it will invalidate the actions
        states = torch.tensor(states).type(dtype)

        # To tensors
        target_pis = torch.tensor(pi).type(dtype)
        greedy_actions = torch.argmax(target_pis, dim=1)

        # Compute losses
        pi_logits = self.pt_actor(states)
        assert pi_logits.shape == target_pis.shape
        loss = F.cross_entropy(pi_logits, greedy_actions)

        # Actor accuracy
        pred_greedy_actions = torch.argmax(pi_logits, dim=1)
        acc = torch.mean((pred_greedy_actions == greedy_actions).type(dtype)).item()

        return loss, acc

    def game_step(self, states, children):
        dtype = self.dtype()
        bsz = len(states)
        size = states[0].shape[-1]

        # To tensors
        states = torch.tensor(states).type(dtype)
        children = children.reshape(bsz, -1, size, size)
        children = torch.tensor(children).type(dtype)

        # Critic Loss
        pred = self.pt_game(states)
        pred = torch.sigmoid(pred)

        critic_loss = 50 * F.mse_loss(pred, children)

        # Predict wins
        return critic_loss

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
        raw_metrics = np.array(raw_metrics, dtype=np.float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_metrics = np.nanmean(raw_metrics, axis=0)
        reduced_metrics = comm.allreduce(mean_metrics, op=MPI.SUM) / world_size

        metrics = ModelMetrics(*reduced_metrics)

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
    def __init__(self, cl=None, ca=None, al=None, aa=None, gl=None):
        self.crit_loss = cl
        self.crit_acc = ca

        self.act_loss = al
        self.act_acc = aa

        self.game_loss = gl

    def __str__(self):
        ret = ''
        if self.crit_loss is not None and not np.isnan(self.crit_loss):
            ret += f'C[{self.crit_acc * 100 :.1f}% {self.crit_loss:.3f}L] '
        if self.act_loss is not None and not np.isnan(self.act_loss):
            ret += f'A[{self.act_acc * 100 :.1f}% {self.act_loss:.3f}L] '
        if self.game_loss is not None and not np.isnan(self.game_loss):
            ret += f'G[{self.game_loss:.3f}L] '

        return ret

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
