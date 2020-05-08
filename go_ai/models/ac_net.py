import numpy as np
import torch
import torch.nn as nn

from go_ai import data
from go_ai.models import RLNet


class ActorCriticNet(RLNet):
    def __init__(self, size):
        action_size = data.GoGame.get_action_size(board_size=size)
        super().__init__(6 + (action_size - 1))

        self.action_size = action_size
        action_template = torch.zeros(1, action_size - 1, size, size, requires_grad=False)
        for a in range(action_size - 1):
            r, c = a // size, a % size
            action_template[0, a, r, c] = 0

        self.action_template = action_template

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

        self.game_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, 6 * (size ** 2 + 1), 1)
        )

    def first(self, states):
        bsz = len(states)
        x = torch.cat([states, self.action_template.expand(bsz, -1, -1, -1)], dim=1)
        x = self.main(x)
        return x

    def forward(self, states):
        x = self.first(states)
        policy_scores = self.act_head(x)
        vals = self.crit_head(x)
        return policy_scores, vals

    # Numpy Calls
    def pt_actor(self, states):
        x = self.first(states)
        return self.act_head(x)

    def pt_game(self, states):
        x = self.first(states)
        return self.game_head(x)

    def pt_critic(self, states):
        x = self.first(states)
        return self.crit_head(x)

    def pt_actor_critic(self, states):
        return self.forward(states)

    # Optimization
    def train_step(self, optimizer, states, actions, reward, children, terminal, wins, pi):
        # Process next states
        bsz = len(children)
        next_states = children[np.arange(bsz), actions]
        next_states = data.batch_random_symmetries(next_states)

        optimizer.zero_grad()

        # Critic
        cl, ca = self.critic_step(next_states, -wins)

        # Actor
        al, aa = self.actor_step(states, pi)

        # Game
        gl = self.game_step(states, children)

        loss = 2 * cl + al + gl
        loss.backward()
        optimizer.step()

        # Return metrics
        return cl.item(), ca, al.item(), aa, gl.item()
