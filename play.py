import gym
import torch

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
    checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.temp)
elif args.agent == 'ac':
    checkpoint_model = actorcritic.ActorCriticNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.checkpath))
    checkpoint_pi = policies.ActorCritic('Checkpoint', checkpoint_model)
print("Loaded model")

opponent_model = actorcritic.ActorCriticNet(args.boardsize)
opponent_model.load_state_dict(torch.load('checkpoints2/block_heur.pt'))
opponent = policies.ActorCritic('Block Heur', opponent_model)

# Play
checkpoint_pi.set_temp(0)
go_env.reset()
wr, _ = utils.parallel_play(go_env, checkpoint_pi, opponent, False, args.evaluations)
print('Winrate: ', wr)