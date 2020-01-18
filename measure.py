import os

import gym
from mpi4py import MPI

import go_ai.policies.actorcritic
import go_ai.policies.baselines
import go_ai.policies.value
from go_ai import measurements, utils

utils.config_log()
args = utils.hyperparameters(MPI.COMM_WORLD)

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
modeldir = 'bin/baselines/'
policy, model = go_ai.policies.baselines.create_policy(args, 'Model', modeldir=modeldir)
utils.log_debug(f"Loaded model {policy} from {modeldir}")

# Directories and files
basedir = os.path.join(modeldir, f'{args.model}{args.boardsize}_plots/')
if not os.path.exists(basedir):
    os.mkdir(basedir)

stats_path = os.path.join(modeldir, f'{args.model}{args.boardsize}_stats.txt')

# Plot stats
if os.path.exists(stats_path):
    measurements.plot_stats(stats_path, basedir)
    utils.log_debug("Plotted ELOs, win rates, losses, and accuracies")

# Plot tree if applicable
if isinstance(policy, go_ai.policies.actorcritic.ActorCritic) or isinstance(policy, go_ai.policies.value.Value):
    go_env.reset()
    measurements.plot_tree(go_env, policy, basedir)
    utils.log_debug(f'Plotted tree')

# Sample trajectory and plot prior qvals
traj_path = os.path.join(basedir, f'heat{policy.temp:.2f}.pdf')
measurements.plot_traj_fig(go_env, policy, traj_path)
utils.log_debug(f"Plotted sample trajectory with temp {args.temp}")
