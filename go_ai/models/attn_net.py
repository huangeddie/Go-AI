import numpy as np
import torch.nn as nn

from go_ai import data
from go_ai.models import RLNet


class AttnNet(RLNet):
    def __init__(self, size):
        super().__init__()
        self.requires_children = True

        self.d_model = 64
        self.nheads = 2
        self.nlayers = 1

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 2, 1),
            nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * size ** 2, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nheads,
                                                   dim_feedforward=self.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.nlayers)

        self.action_size = data.GoGame.get_action_size(board_size=size)

        self.act_head = nn.Linear(self.d_model, 1)

        self.crit_head = nn.Linear(self.d_model, 1)

    def forward(self, states, next_states):
        vals = self.pt_critic(states)
        policy_scores = self.pt_actor(states, next_states)

        return policy_scores, vals

    # Numpy Calls
    def pt_actor(self, _, next_states):
        bsz = len(next_states)
        state_shape = next_states.shape[2:]
        next_states = next_states.view(-1, *state_shape)

        x = self.main(next_states)
        z = self.encoder(x)
        z = z.view(bsz, self.action_size, self.d_model)
        out = self.transformer(z.transpose(0, 1))
        policy_scores = self.act_head(out.transpose(0, 1))
        policy_scores = policy_scores.view(bsz, self.action_size)
        return policy_scores

    def pt_critic(self, states):
        x = self.main(states)
        s = self.encoder(x)
        vals = self.crit_head(s)
        return vals

    def pt_actor_critic(self, states, next_states):
        return self.forward(states, next_states)

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
        al, aa = self.reinforce_step(states, children, actions, wins)

        loss = cl + al
        loss.backward()
        optimizer.step()

        return cl.item(), ca, al.item(), aa
