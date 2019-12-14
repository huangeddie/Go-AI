import gym
import torch
from mpi4py import MPI

import utils
from go_ai import policies, game
from go_ai.models import value, actorcritic

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint')
checkpoint_model.load_state_dict(torch.load(args.checkpath))
print('Loaded checkpoint')

baseline_model, baseline_pi = utils.create_agent(args, 'Baseline', use_base=True)
if baseline_model:
    baseline_model.load_state_dict(torch.load(args.basepath))
print('Loaded baseline')

# Play
comm = MPI.COMM_WORLD
go_env.reset()
wr, _ = utils.parallel_play(comm, go_env, checkpoint_pi, baseline_pi, False, args.evaluations)
print('Winrate: ', wr)