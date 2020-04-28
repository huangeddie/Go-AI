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
        self.assist = True
        self.nblocks = 2
        self.channels = 64

        # Convolutions
        convs = [
            nn.Conv2d(6, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        ]

        for i in range(self.nblocks):
            convs.append(BasicBlock(self.channels, self.channels))

        self.resnet = nn.Sequential(*convs)

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

    def critic_step(self, optimizer, states, wins):
        dtype = next(self.parameters()).type()

        # Preprocess
        states = data.batch_random_symmetries(states)

        # To tensors
        states = torch.tensor(states).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Critic Loss
        optimizer.zero_grad()
        val_logits = self.pt_critic(states)
        vals = torch.tanh(val_logits)

        critic_loss = self.critic_criterion(vals, wins)

        critic_loss.backward()
        optimizer.step()

        # Predict wins
        pred_wins = torch.sign(vals)
        critic_acc = torch.mean((pred_wins == wins).type(dtype))
        return critic_loss.item(), critic_acc.item()

    def reinforce_step(self, optimizer, states, children, actions, wins):
        dtype = next(self.parameters()).type()
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
        optimizer.zero_grad()
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

        actor_loss.backward()
        optimizer.step()

        return actor_loss.item(), 0

    def actor_step(self, optimizer, states, pi):
        dtype = next(self.parameters()).type()

        # Do not augment with random symmetries because it will invalidate the actions
        states = torch.tensor(states).type(dtype)

        # To tensors
        target_pis = torch.tensor(pi).type(dtype)
        greedy_actions = torch.argmax(target_pis, dim=1)

        # Compute losses
        optimizer.zero_grad()
        pi_logits = self.pt_actor(states)
        assert pi_logits.shape == target_pis.shape
        loss = self.actor_criterion(pi_logits, greedy_actions)
        loss.backward()
        optimizer.step()

        # Actor accuracy
        pred_greedy_actions = torch.argmax(pi_logits, dim=1)
        acc = torch.mean((pred_greedy_actions == greedy_actions).type(dtype)).item()

        return loss.item(), acc

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
