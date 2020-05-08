import numpy as np
import torch.nn as nn

from go_ai import data
from go_ai.models import RLNet


class ValueNet(RLNet):
    def __init__(self, size):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return self.convs(x)

    def pt_critic(self, states):
        return self.forward(states)

    def train_step(self, optimizer, _, actions, rewards, children, terminal, wins, pi):
        # Process next states
        bsz = len(children)
        next_states = children[np.arange(bsz), actions]
        next_states = data.batch_random_symmetries(next_states)

        optimizer.zero_grad()

        # Critic
        cl, ca = self.critic_step(next_states, -wins)

        cl.backward()
        optimizer.step()

        return cl.item(), ca, None, None
