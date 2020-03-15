import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

from go_ai.models import BasicBlock, average_model, ModelMetrics
from go_ai import data

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class AttnNet(nn.Module):
    def __init__(self, size, num_blocks=4, channels=32):
        super().__init__()
        self.d_model = 256

        # Convolutions
        convs = [
            nn.Conv2d(6, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]

        for i in range(num_blocks):
            convs.append(BasicBlock(channels, channels))

        encoder = convs + [nn.Conv2d(channels, 2, 1),
                           nn.BatchNorm2d(2), nn.ReLU(),
                           nn.Flatten(),
                           nn.Linear(2 * size ** 2, self.d_model),]

        self.encoder = nn.Sequential(*encoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=self.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.action_size = GoGame.get_action_size(board_size=size)

        self.actor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.CrossEntropyLoss()

    def forward(self, states, next_states):
        vals = self.get_value(states)
        policy_scores = self.get_qs(next_states)

        return policy_scores, vals

    def get_qs(self, next_states):
        bsz = len(next_states)
        state_shape = next_states.shape[2:]
        next_states = next_states.view(-1, *state_shape)
        z = self.encoder(next_states)
        z = z.view(bsz, self.action_size, self.d_model)
        z = z.transpose(0, 1)
        out = self.transformer(z)
        out = out.transpose(0, 1).reshape(-1, self.d_model)
        policy_scores = self.actor(out)
        policy_scores = policy_scores.view(bsz, self.action_size)
        return policy_scores

    def get_value(self, states):
        s = self.encoder(states)
        vals = self.critic(s)
        return vals

    def val_numpy(self, states):
        dtype = next(self.parameters()).type()
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)

            state_vals = self.get_value(tensor_states)
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return vals

    def ac_numpy(self, states):
        """
        :param states: Numpy batch of states
        :return:
        """
        next_states = data.batch_padded_children(states)

        invalid_values = data.batch_invalid_values(states)
        dtype = next(self.parameters()).type()
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)
            tensor_ns = torch.tensor(next_states).type(dtype)
            pi, state_vals = self(tensor_states, tensor_ns)
            pi = pi.detach().cpu().numpy()
            pi += invalid_values
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return pi, vals

    def optimize(self, comm: MPI.Intracomm, batched_data, optimizer):
        dtype = next(self.parameters()).type()

        critic_running_loss = 0
        critic_running_acc = 0

        actor_running_loss = 0
        actor_running_acc = 0

        batches = len(batched_data)
        self.train()
        for states, actions, rewards, _, terminals, wins, target_pis in batched_data:
            next_states = data.batch_padded_children(states)

            states = torch.tensor(states).type(dtype)
            next_states = torch.tensor(next_states).type(dtype)

            wins = torch.tensor(wins[:, np.newaxis]).type(dtype)
            target_pis = torch.tensor(target_pis).type(dtype)
            greedy_actions = torch.argmax(target_pis, dim=1)

            optimizer.zero_grad()
            pi_logits, logits,  = self(states, next_states)
            vals = torch.tanh(logits)
            assert pi_logits.shape == target_pis.shape
            actor_loss = self.actor_criterion(pi_logits, greedy_actions)
            critic_loss = self.critic_criterion(vals, wins)
            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

            pred_greedy_actions = torch.argmax(pi_logits, dim=1)
            actor_running_loss += actor_loss.item()
            actor_running_acc += torch.mean((pred_greedy_actions == greedy_actions).type(dtype)).item()

            pred_wins = torch.sign(vals)
            critic_running_loss += critic_loss.item()
            critic_running_acc += torch.mean((pred_wins == wins).type(dtype)).item()

        # Sync Parameters
        average_model(comm, self)

        world_size = comm.Get_size()
        critic_running_acc = comm.allreduce(critic_running_acc, op=MPI.SUM) / world_size
        critic_running_loss = comm.allreduce(critic_running_loss, op=MPI.SUM) / world_size
        actor_running_acc = comm.allreduce(actor_running_acc, op=MPI.SUM) / world_size
        actor_running_loss = comm.allreduce(actor_running_loss, op=MPI.SUM) / world_size

        metrics = ModelMetrics()
        metrics.crit_acc = critic_running_acc / batches
        metrics.crit_loss = critic_running_loss / batches
        metrics.act_acc = actor_running_acc / batches
        metrics.act_loss = actor_running_loss / batches
        return metrics
