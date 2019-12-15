import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
from tqdm import tqdm

from go_ai import data, montecarlo, parallel
from go_ai.models import BasicBlock, pytorch_ac_to_numpy, average_model, ModelMetrics

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ActorCriticNet(nn.Module):
    def __init__(self, board_size, num_blocks=4, channels=32):
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

        self.shared_convs = nn.Sequential(*convs)

        fc_h = 4 * board_size ** 2
        self.shared_fcs = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU()
        )

        action_size = GoGame.get_action_size(board_size=board_size)
        self.actor = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, 1)
        )

        self.actor_criterion = nn.CrossEntropyLoss()
        self.critic_criterion = nn.MSELoss()

    def forward(self, state):
        x = self.shared_convs(state)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_fcs(x)
        policy_scores = self.actor(x)
        vals = self.critic(x)
        return policy_scores, vals


def parallel_get_qvals(comm, batched_data, val_func):
    batches = len(batched_data)
    world_size = comm.Get_size()
    dividers = batches // world_size
    rank = comm.Get_rank()
    start = rank * dividers
    end = (rank + 1) * dividers if rank < world_size - 1 else batches
    my_batches = batched_data[start: end]
    my_qvals = []
    my_states = []
    for states, actions, next_states, rewards, terminals, wins in my_batches:
        states = data.batch_random_symmetries(states)
        invalid_values = data.batch_invalid_values(states)

        qvals, _ = montecarlo.batchqs_from_valfunc(states, val_func)
        qvals += invalid_values

        my_qvals.append(qvals)
        my_states.append(states)

    all_data = comm.allgather((my_qvals, my_states))
    all_qvals = []
    all_states = []

    for qvals, states in all_data:
        all_qvals.extend(qvals)
        all_states.extend(states)

    return all_qvals, all_states


def optimize(comm: MPI.Intracomm, model: ActorCriticNet, batched_data, optimizer):
    dtype = next(model.parameters()).type()

    critic_running_loss = 0
    critic_running_acc = 0
    model.train()
    for states, actions, next_states, rewards, terminals, wins in batched_data:
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(dtype)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(dtype)

        optimizer.zero_grad()
        _, logits = model(states)
        vals = torch.tanh(logits)
        loss = model.critic_criterion(vals, wins)
        loss.backward()
        optimizer.step()

        pred_wins = torch.sign(vals)
        critic_running_loss += loss.item()
        critic_running_acc += torch.mean((pred_wins == wins).type(dtype)).item()
    parallel.parallel_err(comm, 'Optimized Critic')
    # Sync Parameters
    average_model(comm, model)

    pi_func, val_func = pytorch_ac_to_numpy(model)
    qvals, states = parallel_get_qvals(comm, batched_data, val_func)
    parallel.parallel_err(comm, 'Calculated QVals')

    actor_running_loss = 0
    actor_running_acc = 0
    batches = len(batched_data)

    model.train()
    for qvals, states in zip(qvals, states):
        # Augment
        greedy_actions = torch.from_numpy(np.argmax(qvals, axis=1))

        optimizer.zero_grad()
        states = torch.from_numpy(states).type(dtype)
        policy_scores, _ = model(states)

        loss = model.actor_criterion(policy_scores, greedy_actions)
        loss.backward()
        optimizer.step()

        pred_actions = torch.argmax(policy_scores, dim=1)
        actor_running_loss += loss.item()
        actor_running_acc += torch.mean((pred_actions == greedy_actions).type(dtype)).item()

    parallel.parallel_err(comm, 'Optimized Actor')

    # Sync Parameters
    average_model(comm, model)

    metrics = ModelMetrics()
    metrics.crit_acc = critic_running_acc / batches
    metrics.crit_loss = critic_running_loss / batches
    metrics.act_acc = actor_running_acc / batches
    metrics.act_loss = actor_running_loss / batches
    return metrics
