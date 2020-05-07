import os

import gym

from go_ai.policies import actorcritic, baselines, value, qval
import go_ai.search.plot
from go_ai import measurements, utils

utils.config_log()
args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.size)

if args.baseline:
    outdir = 'bin/baselines/'
else:
    outdir = args.customdir

# Policies
policy, model = baselines.create_policy(args, 'Model')

# Directories and files
plotsdir = os.path.join(outdir, f'{args.model}{args.size}_plots/')
if not os.path.exists(plotsdir):
    os.mkdir(plotsdir)

stats_path = os.path.join(outdir, f'{args.model}{args.size}_stats.txt')

# Plot stats
if os.path.exists(stats_path):
    measurements.plot_stats(stats_path, plotsdir)
    utils.log_debug("Plotted ELOs, win rates, losses, and accuracies")

# Plot tree if applicable
if isinstance(policy, actorcritic.ActorCritic) or isinstance(policy, value.Value):
    blacks = []
    whites = []
    utils.log_debug(f'Plotting tree...')
    go_ai.search.plot.plot_tree(go_env, policy, plotsdir, [blacks, whites])
    utils.log_debug(f'Plotted tree')
    
# Plot Go understanding if applicable
if isinstance(policy, qval.QVal):
    go_path = os.path.join(plotsdir, 'go.pdf')
    utils.log_debug(f'Plotting Go understanding...')
    measurements.plot_go_understanding(go_env, policy, go_path)
    utils.log_debug(f'Plotted Go understanding')

# Sample trajectory and plot prior qvals
traj_path = os.path.join(plotsdir, f'heat{policy.temp:.2f}.pdf')
measurements.plot_traj_fig(go_env, policy, traj_path)
utils.log_debug(f"Plotted sample trajectory {traj_path}")
