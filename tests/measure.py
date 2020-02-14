import os

import gym

import go_ai.policies.actorcritic
import go_ai.policies.baselines
import go_ai.policies.value
import go_ai.search.plot
from go_ai import measurements, utils

utils.config_log()
args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.size)

customdir = 'bin/baselines/'

# Policies
policy, model = go_ai.policies.baselines.create_policy(args, 'Model')

# Directories and files
plotsdir = os.path.join(customdir, f'{args.model}{args.size}_plots/')
if not os.path.exists(plotsdir):
    os.mkdir(plotsdir)

stats_path = os.path.join(customdir, f'{args.model}{args.size}_stats.txt')

# Plot stats
if os.path.exists(stats_path):
    measurements.plot_stats(stats_path, plotsdir)
    utils.log_debug("Plotted ELOs, win rates, losses, and accuracies")

# Plot tree if applicable
if False and (isinstance(policy, go_ai.policies.actorcritic.ActorCritic) or isinstance(policy, go_ai.policies.value.Value)):
    black_rows = []
    black_cols = []
    white_rows = []
    white_cols = []
    blacks = list(zip(black_rows, black_cols))
    whites = list(zip(white_rows, white_cols))
    utils.log_debug(f'Plotting tree...')
    go_ai.search.plot.plot_tree(go_env, policy, plotsdir, [blacks, whites])
    utils.log_debug(f'Plotted tree')

# Sample trajectory and plot prior qvals
traj_path = os.path.join(plotsdir, f'heat{policy.temp:.2f}.pdf')
measurements.plot_traj_fig(go_env, policy, traj_path)
utils.log_debug(f"Plotted sample trajectory {traj_path}")
