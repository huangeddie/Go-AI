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
checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, 0, args.temp)
print("Loaded model")

# Sample trajectory and plot prior qvals
metrics.plot_traj_fig(go_env, checkpoint_pi, f'episodes/atraj_{checkpoint_pi.temp:.5f}.pdf')
print("Plotted sample trajectory")

# Play
checkpoint_pi.set_temp(0)
checkpoint_pi.num_searches = args.mcts
go_env.reset()
game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
