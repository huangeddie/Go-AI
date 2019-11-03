import gym
import torch

import utils
from go_ai import policies, game
from go_ai.models import value_models, actorcritic_model

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
if args.agent == 'mcts':
    checkpoint_model = value_models.ValueNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.check_path))
    checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.temp)
elif args.agent == 'ac':
    checkpoint_model = actorcritic_model.ActorCriticNet(args.boardsize)
    checkpoint_model.load_state_dict(torch.load(args.check_path))
    checkpoint_pi = policies.ActorCritic('Checkpoint', checkpoint_model, args.temp)
print("Loaded model")

# Play
checkpoint_pi.set_temp(0)
go_env.reset()
game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
