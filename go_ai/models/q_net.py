import numpy as np
import torch.nn as nn

from go_ai import data
from go_ai.models import RLNet


class QNet(RLNet):
    def __init__(self, size):
        super().__init__(7)

        self.critic = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.go = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, 6, 3, 1, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return self.critic(x), self.go(x)

    def pt_critic(self, states):
        x = self.main(states)
        return self.critic(x)

    def pt_game(self, state_actions):
        x = self.main(state_actions)
        return self.go(x)

    def train_step(self, optimizer, states, actions, rewards, children, terminal, wins, pi):
        # Critic
        state_actions = data.batch_combine_state_actions(states, actions)
        cl, ca = self.critic_step(optimizer, state_actions, wins)

        # Game
        bsz = len(children)
        next_states = children[np.arange(bsz), actions]
        gl = self.game_step(optimizer, states, actions, next_states)

        return cl, ca, None, None, gl
