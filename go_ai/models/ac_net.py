import numpy as np
import torch.nn as nn

from go_ai import data
from go_ai.models import RLNet


class ActorCriticNet(RLNet):
    def __init__(self, size):
        super().__init__()
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

    def forward(self, state):
        x = self.main(state)
        policy_scores = self.act_head(x)
        vals = self.crit_head(x)
        return policy_scores, vals

    # Numpy Calls
    def pt_actor(self, states):
        x = self.main(states)
        return self.act_head(x)

    def pt_critic(self, states):
        x = self.main(states)
        return self.crit_head(x)

    def pt_actor_critic(self, states):
        return self.forward(states)

    # Optimization
    def train_step(self, optimizer, states, actions, reward, children, terminal, wins, pi):
        # Process next states
        bsz = len(children)
        next_states = children[np.arange(bsz), actions]
        next_states = data.batch_random_symmetries(next_states)
        # Critic
        cl, ca = self.critic_step(optimizer, next_states, -wins)

        # Actor
        al, aa = self.actor_step(optimizer, states, pi)

        # Return metrics
        return cl, ca, al, aa
