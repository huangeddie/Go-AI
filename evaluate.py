import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from mpi4py import MPI

from go_ai import measurements, utils, policies
from go_ai.montecarlo import tree

args = utils.hyperparameters(MPI.COMM_WORLD)

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
modeldir = 'bin/checkpoints/2019-12-28/'
model, policy = utils.create_model(args, 'Checkpoint', checkdir=modeldir)
print(f"Loaded model {policy} from {modeldir}")

basedir = os.path.join(modeldir, f'{args.model}{args.boardsize}_plots/')
if not os.path.exists(basedir):
    os.mkdir(basedir)

stats_path = os.path.join(modeldir, f'{args.model}{args.boardsize}_stats.txt')


def convert_to_secs(time_str):
    dur = time.strptime(time_str, '%H:%M:%S')
    secs = 3600 * dur.tm_hour + 60 * dur.tm_min + dur.tm_sec
    return secs


def convert_to_hours(time_str):
    return convert_to_secs(time_str) / 3600


# Plots
if os.path.exists(stats_path):
    df = pd.read_csv(stats_path, sep='\t')

    df['HOURS'] = df['TIME'].map(convert_to_hours)

    # Elo
    # New checkpoints
    check_elos = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            prev_elo = 0
        else:
            prev_elo = check_elos[i - 1]
        wr = df['C_WR'].values[i] / 100
        delta = 400 * (2 * wr - 1)
        check_elos[i] = prev_elo + delta
    plt.title('ELO Score')
    plt.plot(df['HOURS'], check_elos)
    plt.xlabel("Hours")
    plt.ylabel("ELO")
    plt.savefig(os.path.join(basedir, 'elos.pdf'))
    plt.close()
    print("Plotted ELOs")

    # Win rate against random and greedy
    plt.figure()
    plt.title('Winrates against Baseline Models')
    plt.plot(df['HOURS'], df['R_WR'])
    plt.plot(df['HOURS'], df['G_WR'])
    plt.xlabel('Hours')
    plt.ylabel('Winrate')
    plt.legend(['Random', 'Greedy'])
    plt.savefig(os.path.join(basedir, 'winrates.pdf'))
    plt.close()
    print("Plotted win rates")

# Plot tree if applicable
if isinstance(policy, policies.ActorCritic):
    go_env.reset()
    root = policy.get_tree(go_env)
    imgdir = os.path.join(basedir, 'node_imgs/')
    imgdir = os.path.abspath(imgdir)
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    graph = tree.get_graph(root, imgdir)
    graph.render(os.path.join(basedir, 'tree'))
    print(f'Plotted tree')

# Sample trajectory and plot prior qvals
traj_path = os.path.join(basedir, f'heat{policy.temp:.2f}.pdf')
measurements.plot_traj_fig(go_env, policy, traj_path)
print(f"Plotted sample trajectory with temp {args.temp}")
