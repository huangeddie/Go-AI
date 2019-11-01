import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from go_ai import data, montecarlo

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ActorCriticNet(nn.Module):
    def __init__(self, board_size):
        super(ActorCriticNet, self).__init__()
        self.shared_convs = nn.Sequential(
            nn.Conv2d(GoVars.NUM_CHNLS, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.shared_fcs = nn.Sequential(
            nn.Linear(4 * board_size ** 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        action_size = GoGame.get_action_size(board_size=board_size)
        self.actor = nn.Sequential(
            nn.Linear(256, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.actor_criterion = nn.CrossEntropyLoss()
        self.critic_criterion = nn.BCEWithLogitsLoss()

    def forward(self, state):
        invalid_values = data.batch_invalid_values(state)
        x = self.shared_convs(state)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_fcs(x)
        policy_scores = self.actor(x)
        policy_scores += invalid_values
        vals = self.critic(x)
        return policy_scores, vals

    def get_probs(self, policy_scores, state, temp):
        valid_moves = data.batch_valid_moves(state)
        return montecarlo.exp_temp(policy_scores, temp, valid_moves)


def optimize(model, replay_data, optimizer, batch_size):
    N = len(replay_data[0])
    for component in replay_data:
        assert len(component) == N

    batched_data = [np.array_split(component, N // batch_size) for component in replay_data]
    batched_data = list(zip(*batched_data))

    model.train()
    critic_running_loss = 0
    critic_running_acc = 0
    pbar = tqdm(batched_data, desc="Optimizing critic", leave=False)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(torch.FloatTensor)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(torch.FloatTensor)

        optimizer.zero_grad()
        _, vals = model(states)
        pred_wins = (torch.sigmoid(vals) > 0.5).type(vals.dtype)
        loss = model.critic_criterion(vals, wins)
        loss.backward()
        optimizer.step()

        critic_running_loss += loss.item()
        critic_running_acc += torch.mean((pred_wins == wins).type(wins.dtype)).item()

        pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * critic_running_acc / i, critic_running_loss / i))
    pbar.close()

    actor_running_loss = 0
    actor_running_acc = 0
    batches = 0
    pbar = tqdm(batched_data, desc="Optimizing actor", leave=False)
    for i, (states, actions, next_states, rewards, terminals, wins) in enumerate(pbar, 1):
        # Augment
        states = data.batch_random_symmetries(states)

        states = torch.from_numpy(states).type(torch.FloatTensor)
        wins = torch.from_numpy(wins[:, np.newaxis]).type(torch.FloatTensor)

        optimizer.zero_grad()
        policy_scores, _ = model(states)
        pred_actions = torch.argmax(policy_scores, dim=1)
        qvals = montecarlo.qval_from_stateval(states, lambda s: model(s)[1])
        greedy_actions = torch.from_numpy(np.argmax(qvals, axis=1)).type(torch.FloatTensor)
        loss = model.actor_criterion(policy_scores, greedy_actions)
        loss.backward()
        optimizer.step()

        actor_running_loss += loss.item()
        actor_running_acc += torch.mean((pred_actions == greedy_actions).type(wins.dtype)).item()
        batches = i

        pbar.set_postfix_str("{:.1f}%, {:.3f}L".format(100 * actor_running_acc / i, actor_running_loss / i))
    pbar.close()
    
    return (critic_running_acc / batches, critic_running_loss / batches,
        actor_running_acc / batches, actor_running_loss / batches)
