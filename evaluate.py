import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from go_ai import measurements, utils

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_dir = 'bin/checkpoints/2019-12-11/'
checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint', checkpoint=checkpoint_dir)
print(f"Loaded model {checkpoint_pi} from {args.savedir}")

stats_path = os.path.join(checkpoint_dir, 'stats.txt')


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
    checks = df[df['C_WR'] > 55]
    check_elos = np.zeros(len(checks))
    for i in range(len(checks)):
        if i == 0:
            prev_elo = 0
        else:
            prev_elo = check_elos[i - 1]
        wr = checks['C_WR'].values[i] / 100
        check_elos[i] = prev_elo + 400 * (2 * wr - 1)
    plt.title('ELO Score')
    plt.plot(checks['HOURS'], check_elos)
    plt.xlabel("Hours")
    plt.ylabel("ELO")
    plt.savefig(os.path.join(checkpoint_dir, 'elos.pdf'))
    plt.close()

    # Win rate against random and greedy
    plt.figure()
    plt.title('Winrates against Baseline Models')
    plt.plot(df['HOURS'], df['R_WR'])
    plt.plot(df['HOURS'], df['G_WR'])
    plt.xlabel('Hours')
    plt.ylabel('Winrate')
    plt.legend(['Random', 'Greedy'])
    plt.savefig(os.path.join(checkpoint_dir, 'winrates.pdf'))
    plt.close()

    print("Made plots")

# Sample trajectory and plot prior qvals
measurements.plot_traj_fig(go_env, checkpoint_pi, os.path.join(checkpoint_dir, f'atraj_{checkpoint_pi.temp:.2f}.pdf'))
print(f"Plotted sample trajectory with temp {args.temp}")
