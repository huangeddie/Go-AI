import numpy as np
import torch
import torch.nn as nn

from go_ai import data
from go_ai.models import BasicBlock, RLNet


class AttnNet(RLNet):
    def __init__(self, size):
        super().__init__()
        self.requires_children = True

        self.d_model = 64
        self.nheads = 2
        self.nlayers = 1

        # Convolutions
        convs = [
            nn.Conv2d(6, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        ]

        for i in range(self.nblocks):
            convs.append(BasicBlock(self.channels, self.channels))

        encoder = convs + [nn.Conv2d(self.channels, 2, 1),
                           nn.BatchNorm2d(2), nn.ReLU(),
                           nn.Flatten(),
                           nn.Linear(2 * size ** 2, self.d_model)]

        self.encoder = nn.Sequential(*encoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nheads,
                                                   dim_feedforward=self.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.nlayers)

        self.action_size = data.GoGame.get_action_size(board_size=size)

        self.act_head = nn.Linear(self.d_model, 1)

        self.crit_head = nn.Linear(self.d_model, 1)

        self.critic_criterion = nn.MSELoss()

    def forward(self, states, next_states):
        vals = self.pt_critic(states)
        policy_scores = self.pt_actor(states, next_states)

        return policy_scores, vals

    # Numpy Calls
    def pt_actor(self, _, next_states):
        bsz = len(next_states)
        state_shape = next_states.shape[2:]
        next_states = next_states.view(-1, *state_shape)
        z = self.encoder(next_states)
        z = z.view(bsz, self.action_size, self.d_model)
        out = self.transformer(z.transpose(0, 1))
        policy_scores = self.act_head(out.transpose(0, 1))
        policy_scores = policy_scores.view(bsz, self.action_size)
        return policy_scores

    def pt_critic(self, states):
        s = self.encoder(states)
        vals = self.crit_head(s)
        return vals

    def pt_actor_critic(self, states, next_states):
        return self.forward(states, next_states)

    # Optimization
    def train_step(self, optimizer, states, actions, reward, children, terminal, wins, pi):
        cl, ca = self.critic_step(optimizer, states, wins)
        al, aa = self.actor_step(optimizer, states, children, actions, wins)
        return cl, ca, al, aa

    def critic_step(self, optimizer, states, wins):
        dtype = next(self.parameters()).type()

        # Preprocess
        states = data.batch_random_symmetries(states)

        # To tensors
        states = torch.tensor(states).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Critic Loss
        optimizer.zero_grad()
        val_logits = self.pt_critic(states)
        vals = torch.tanh(val_logits)

        critic_loss = self.critic_criterion(vals, wins)

        critic_loss.backward()
        optimizer.step()

        # Predict wins
        pred_wins = torch.sign(vals)
        critic_acc = torch.mean((pred_wins == wins).type(dtype))
        return critic_loss.item(), critic_acc.item()

    def actor_step(self, optimizer, states, children, actions, wins):
        dtype = next(self.parameters()).type()
        bsz = len(states)

        # Preprocess
        states = data.batch_random_symmetries(states)

        # To tensors
        states = torch.tensor(states).type(dtype)
        children = torch.tensor(children).type(dtype)
        wins = torch.tensor(wins[:, np.newaxis]).type(dtype)

        # Value for baseline
        with torch.no_grad():
            val_logits = self.pt_critic(states)
            vals = torch.tanh(val_logits)

        # Forward pass
        optimizer.zero_grad()
        pi_logits = self.pt_actor(states, children)

        # Log probability of taken actions
        logpi_logits = torch.log_softmax(pi_logits, dim=1)
        log_pis = logpi_logits[torch.arange(bsz), actions].reshape(-1, 1)

        # Policy Gradient
        assert vals.shape == wins.shape
        advantages = wins - vals
        assert log_pis.shape == advantages.shape
        expected_reward = log_pis * advantages
        actor_loss = -torch.mean(expected_reward)

        actor_loss.backward()
        optimizer.step()

        return actor_loss.item(), 0
