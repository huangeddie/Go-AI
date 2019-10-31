import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import utils
from go_ai import policies, metrics
from go_ai.models import value_models

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model = value_models.ValueNet(args.boardsize)
checkpoint_model.load_state_dict(torch.load(args.check_path))
checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.temp, args.tempsteps)
print("Loaded model")

# Elo
if os.path.exists('checkpoints/stats.txt'):
    rand_elo = 0
    greed_elo = 0
    df = pd.read_csv('checkpoints/stats.txt', sep='\t')

    # New checkpoints
    checks = df[df['C_WR'] > 55]
    check_elos = np.zeros(len(checks))
    for i in range(len(checks)):
        if i == 0:
            prev_elo = 0
        else:
            prev_elo = check_elos[i - 1]
        wr = checks['C_WR'].values[i] / 100
        check_elos[i] = prev_elo + 400 * (2 * wr - 1)
    plt.title('Crude ELOs')
    seconds = []
    for time_str in checks['TIME']:
        dur = time.strptime(time_str, '%H:%M:%S')
        secs = 3600 * dur.tm_hour + 60 * dur.tm_min + dur.tm_sec
        seconds.append(secs)
    seconds = np.array(seconds)
    plt.plot(seconds / 3600, check_elos)
    plt.xlabel("Hours")
    plt.ylabel("ELO")
    plt.savefig('checkpoints/elos.pdf')
    print("Plotted Elos")

# Sample trajectory and plot prior qvals
metrics.plot_traj_fig(go_env, checkpoint_pi, f'checkpoints/atraj_{checkpoint_pi.temp:.4f}.pdf')
print("Plotted sample trajectory")
