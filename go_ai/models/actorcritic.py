import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from go_ai import data, montecarlo, policies

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ActorCriticNet(nn.Module):
    def __init__(self, board_size):
        super(ActorCriticNet, self).__init__()
        self.shared_convs = nn.Sequential(
            nn.Conv2d(GoVars.NUM_CHNLS, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.shared_fcs = nn.Sequential(
            nn.Linear(board_size ** 2, board_size ** 2),
            nn.BatchNorm1d(board_size ** 2),
            nn.ReLU(),
        )

        action_size = GoGame.get_action_size(board_size=board_size)
        self.actor = nn.Sequential(
            nn.Linear(board_size ** 2, board_size ** 2),
            nn.BatchNorm1d(board_size ** 2),
            nn.ReLU(),
            nn.Linear(board_size ** 2, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(board_size ** 2, board_size ** 2),
            nn.BatchNorm1d(board_size ** 2),
            nn.ReLU(),
            nn.Linear(board_size ** 2, 1),
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


class CriticWrapper(nn.Module):
    def __init__(self, net):
        super(CriticWrapper, self).__init__()
        self.net = net

    def forward(self, state):
        return self.net(state)[1]

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()


def optimize(model, batched_data, optimizer):
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

    val_func = policies.pytorch_to_numpy(CriticWrapper(model))

    actor_running_loss = 0
    actor_running_acc = 0
    batches = 0
    pbar = tqdm(batched_data, desc="Optimizing actor", leave=True)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)
        invalid_values = data.batch_invalid_values(states)

        states = torch.from_numpy(states).type(dtype)

        optimizer.zero_grad()
        policy_scores, _ = model(states)

        qvals = montecarlo.qs_from_stateval(states, val_func)[0]
        qvals += invalid_values
        greedy_actions = torch.from_numpy(np.argmax(qvals, axis=1)).type(torch.LongTensor)

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

    return (critic_running_acc / batches, critic_running_loss / batches,
            actor_running_acc / batches, actor_running_loss / batches)
