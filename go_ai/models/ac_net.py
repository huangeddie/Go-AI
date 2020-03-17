import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

from go_ai import data
from go_ai.models import BasicBlock, average_model, ModelMetrics

gymgo = gym.make('gym_go:go-v0', size=0)
GoGame = gymgo.gogame
GoVars = gymgo.govars


class ActorCriticNet(nn.Module):
    def __init__(self, size, num_blocks=4, channels=32):
        super().__init__()
        # Convolutions
        convs = [
            nn.Conv2d(6, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]

        for i in range(num_blocks):
            convs.append(BasicBlock(channels, channels))

        self.shared_convs = nn.Sequential(*convs)

        action_size = GoGame.get_action_size(board_size=size)

        self.actor = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * size ** 2, action_size),
        )

        self.critic = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
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
        policy_scores = self.actor(x)
        vals = self.critic(x)
        return policy_scores, vals

    def val_numpy(self, states):
        dtype = next(self.parameters()).type()
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)
            z = self.shared_convs(tensor_states)
            state_vals = self.critic(z)
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = 100 * GoGame.get_winning(state)

        return vals

    def ac_numpy(self, states):
        invalid_values = data.batch_invalid_values(states)
        dtype = next(self.parameters()).type()
        self.eval()
        with torch.no_grad():
            tensor_states = torch.tensor(states).type(dtype)
            pi, state_vals = self(tensor_states)
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
        for states, _, _, _, _, wins, target_pis in batched_data:
            states = torch.tensor(states).type(dtype)

            wins = torch.tensor(wins[:, np.newaxis]).type(dtype)
            target_pis = torch.tensor(target_pis).type(dtype)
            greedy_actions = torch.argmax(target_pis, dim=1)

            optimizer.zero_grad()
            pi_logits, logits = self(states)
            vals = torch.tanh(logits)
            assert pi_logits.shape == target_pis.shape
            actor_loss = self.actor_criterion(pi_logits, greedy_actions)
            critic_loss = self.critic_criterion(vals, wins)
            loss = 0.5 * actor_loss + critic_loss
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
