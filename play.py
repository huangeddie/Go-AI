import gym
import torch

import utils
from go_ai import policies, game, metrics
from go_ai.models import value_models

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model = value_models.ValueNet(args.boardsize)
checkpoint_model.load_state_dict(torch.load(args.check_path))
checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, 0, 0)

# Sample trajectory
metrics.plot_traj_fig(go_env, checkpoint_pi, 'episodes/atraj.pdf')

# Play
game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
