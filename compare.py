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
if args.agent == 'mcts':
    checkpoint_model = value.ValueNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.checkpath))
    checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, 82, 0)
elif args.agent == 'ac':
    checkpoint_model = actorcritic.ActorCriticNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.checkpath))
    checkpoint_pi = policies.ActorCritic('Checkpoint', checkpoint_model)
elif args.agent == 'mcts-ac':
    checkpoint_model = actorcritic.ActorCriticNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.checkpath))
    checkpoint_pi = policies.MCTSActorCritic('Checkpoint', checkpoint_model, args.branches, args.depth)
print('Loaded checkpoint')

if args.baseagent == 'mcts':
    baseline_model = value.ValueNet(args.boardsize)
    baseline_model.load_state_dict(torch.load(args.basepath))
    baseline_pi = policies.MCTS('Baseline', baseline_model, 82, 0)
elif args.baseagent == 'ac':
    baseline_model = actorcritic.ActorCriticNet(args.boardsize)
    baseline_model.load_state_dict(torch.load(args.basepath))
    baseline_pi = policies.ActorCritic('Baseline', baseline_model)
elif args.baseagent == 'mcts-ac':
    baseline_model = actorcritic.ActorCriticNet(args.boardsize)
    baseline_model.load_state_dict(torch.load(args.basepath))
    baseline_pi = policies.MCTSActorCritic('Baseline', baseline_model, args.branches, args.depth)
elif args.baseagent == 'rand':
    baseline_pi = policies.RAND_PI
elif args.baseagent == 'greedy':
    baseline_pi = policies.GREEDY_PI
print('Loaded baseline')

# Play
comm = MPI.COMM_WORLD
go_env.reset()
wr, _ = utils.parallel_play(comm, go_env, checkpoint_pi, baseline_pi, False, args.evaluations)
print('Winrate: ', wr)