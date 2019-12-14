import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
from tqdm import tqdm

import go_ai.models
from go_ai import data, montecarlo
from go_ai.models import BasicBlock

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars

class ActorCriticNet(nn.Module):
    def __init__(self, board_size, num_blocks=4, channels=64):
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
            nn.ReLU(),
        )

        action_size = GoGame.get_action_size(board_size=board_size)
        self.actor = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, 1),
            nn.Tanh(),
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


class ActorCriticWrapper(nn.Module):
    def __init__(self, net, mode):
        super(ActorCriticWrapper, self).__init__()
        self.net = net
        if mode == 'actor':
            self.index = 0
        elif mode == 'critic':
            self.index = 1
        else:
            raise RuntimeError('Invalid mode specified for ActorCriticWrapper')

    def forward(self, state):
        return self.net(state)[self.index]

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()


def optimize(comm: MPI.Intracomm, model, batched_data, optimizer):
    model.train()
    dtype = next(model.parameters()).type()
    critic_running_loss = 0
    critic_running_acc = 0
    pbar = tqdm(batched_data, desc="Optimizing critic", leave=True)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(dtype)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(dtype)

        optimizer.zero_grad()
        _, vals = model(states)
        loss = model.critic_criterion(vals, wins)
        loss.backward()
        optimizer.step()

        pred_wins = torch.sign(vals)
        critic_running_loss += loss.item()
        critic_running_acc += torch.mean((pred_wins == wins).type(dtype)).item()

        pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * critic_running_acc / i, critic_running_loss / i))

    val_func = go_ai.models.pytorch_to_numpy(ActorCriticWrapper(model, 'critic'), scale=1)

    actor_running_loss = 0
    actor_running_acc = 0
    batches = 0
    pbar = tqdm(batched_data, desc="Optimizing actor", leave=True)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)
        invalid_values = data.batch_invalid_values(states)

        qvals = montecarlo.batchqs_from_valfunc(states, val_func)[0]
        qvals += invalid_values
        greedy_actions = torch.from_numpy(np.argmax(qvals, axis=1)).type(torch.LongTensor)

        states = torch.from_numpy(states).type(dtype)

        optimizer.zero_grad()
        policy_scores, _ = model(states)

        loss = model.actor_criterion(policy_scores, greedy_actions)
        if (loss > 1000).any():
            print('Big loss!')
            print('greedy_actions: ', greedy_actions)
            greedy_scores = []
            for j, a in enumerate(greedy_actions):
                greedy_scores.append(policy_scores[j, a].item())
            print('policy_scores[greedy]: ', greedy_scores)
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
