import numpy as np
import torch
import torch.nn as nn

from go_ai import data
from go_ai.models import BasicBlock, RLNet


class ActorCriticNet(RLNet):
    def __init__(self, size):
        super().__init__()
        # Convolutions
        convs = [
            nn.Conv2d(6, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        ]

        for i in range(self.nblocks):
            convs.append(BasicBlock(self.channels, self.channels))

        self.shared_convs = nn.Sequential(*convs)

        action_size = data.GoGame.get_action_size(board_size=size)

        self.act_head = nn.Sequential(
            nn.Conv2d(self.channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * size ** 2, action_size),
        )

        self.crit_head = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.CrossEntropyLoss()

    def forward(self, state):
        x = self.shared_convs(state)
        policy_scores = self.act_head(x)
        vals = self.crit_head(x)
        return policy_scores, vals

    # Numpy Calls
    def pt_actor(self, states):
        x = self.shared_convs(states)
        return self.act_head(x)

    def pt_critic(self, states):
        x = self.shared_convs(states)
        return self.crit_head(x)

    def pt_actor_critic(self, states):
        return self.forward(states)

    # Optimization
    def train_step(self, optimizer, states, actions, reward, children, terminal, wins, pi):
        dtype = next(self.parameters()).type()

        # Preprocess states
        states = data.batch_random_symmetries(states)
        states = torch.tensor(states).type(dtype)

        # To tensors
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)
        target_pis = torch.tensor(pi).type(dtype)
        greedy_actions = torch.argmax(target_pis, dim=1)

        # Compute losses
        optimizer.zero_grad()
        pi_logits, logits = self(states)
        vals = torch.tanh(logits)
        assert pi_logits.shape == target_pis.shape
        actor_loss = self.actor_criterion(pi_logits, greedy_actions)
        critic_loss = self.critic_criterion(vals, wins)
        loss = actor_loss + critic_loss
        loss.backward()
        optimizer.step()

        # Actor accuracy
        pred_greedy_actions = torch.argmax(pi_logits, dim=1)
        actor_acc = torch.mean((pred_greedy_actions == greedy_actions).type(dtype)).item()

        # Critic accuracy
        pred_wins = torch.sign(vals)
        critic_acc = torch.mean((pred_wins == wins).type(dtype)).item()

        # Return metrics
        return critic_loss.item(), critic_acc, actor_loss.item(), actor_acc
