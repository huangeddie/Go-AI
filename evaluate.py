import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import utils
from go_ai import metrics, game

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint')
checkpoint_model.load_state_dict(torch.load(args.checkpath))
print(f"Loaded model {checkpoint_pi} from {args.checkpath}")

# Elo
if os.path.exists('bin/stats.txt'):
    rand_elo = 0
    greed_elo = 0
    df = pd.read_csv('bin/stats.txt', sep='\t')

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
    plt.savefig('bin/elos.pdf')
    print("Plotted Elos")

# Sample trajectory and plot prior qvals
metrics.plot_traj_fig(go_env, checkpoint_pi, f'bin/atraj_{checkpoint_pi.temp:.2f}.pdf')
print(f"Plotted sample trajectory with temp {args.temp}")

# Play against baseline model
baseline_model, baseline_pi = utils.create_agent(args, 'Baseline', use_base=True)
if baseline_model:
    baseline_model.load_state_dict(torch.load(args.basepath))
print('Loaded baseline')

# Play
go_env.reset()
wr, _, _ = game.play_games(go_env, checkpoint_pi, baseline_pi, False, args.evaluations)
print('Winrate: ', wr)
