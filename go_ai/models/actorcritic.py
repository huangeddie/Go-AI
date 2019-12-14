import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
from tqdm import tqdm

import go_ai.models
from go_ai.models import pytorch_ac_to_numpy
from go_ai import data, montecarlo
from go_ai.models import BasicBlock

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
        invalid_values = data.batch_invalid_values(state)
        x = self.shared_convs(state)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_fcs(x)
        policy_scores = self.actor(x)
        policy_scores += invalid_values
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
    for states, actions, next_states, rewards, terminals, wins in tqdm(my_batches, desc='Getting QVals in Parallel'):
        states = data.batch_random_symmetries(states)
        invalid_values = data.batch_invalid_values(states)

        qvals = montecarlo.batchqs_from_valfunc(states, val_func)[0]
        qvals += invalid_values

        my_qvals.append(qvals)
        my_states.append(states)
    comm.Barrier()
    all_data = comm.gather((my_qvals, my_states), 0)
    all_qvals = []
    all_states = []

    if rank == 0:
        for qvals, states in all_data:
            all_qvals.extend(qvals)
            all_states.extend(states)

    return all_qvals, all_states

def optimize(comm: MPI.Intracomm, model: ActorCriticNet, batched_data, optimizer):
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    dtype = next(model.parameters()).type()

    critic_running_loss = 0
    critic_running_acc = 0
    if rank == 0:
        model.train()
        pbar = tqdm(batched_data, desc="Optimizing critic", leave=True)
        for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
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

            pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * critic_running_acc / i, critic_running_loss / i))

    # Sync Parameters
    comm.Barrier()
    for params in model.parameters():
        params.data = comm.allreduce(params.data, op=MPI.SUM) / world_size

    pi_func, val_func = pytorch_ac_to_numpy(model)
    qvals, states = parallel_get_qvals(comm, batched_data, val_func)

    actor_running_loss = 0
    actor_running_acc = 0
    batches = len(batched_data)
    if rank == 0:
        model.train()
        pbar = tqdm(list(zip(qvals, states)), desc="Optimizing actor", leave=True)
        for i, (qvals, states) in enumerate(pbar, 1):
            # Augment
            greedy_actions = torch.from_numpy(np.argmax(qvals, axis=1)).type(torch.LongTensor)

            optimizer.zero_grad()
            states = torch.from_numpy(states).type(dtype)
            policy_scores, _ = model(states)

            loss = model.actor_criterion(policy_scores, greedy_actions)
            loss.backward()
            optimizer.step()

            pred_actions = torch.argmax(policy_scores, dim=1)
            actor_running_loss += loss.item()
            actor_running_acc += torch.mean((pred_actions == greedy_actions).type(dtype)).item()
            batches = i

            pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * actor_running_acc / i, actor_running_loss / i))

    metrics = go_ai.models.ModelMetrics()
    metrics.crit_acc = critic_running_acc / batches
    metrics.crit_loss = critic_running_loss / batches
    metrics.act_acc = actor_running_acc / batches
    metrics.act_loss = actor_running_loss / batches
    return metrics
