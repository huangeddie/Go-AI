import os
import time

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import go_ai.policies
import go_ai.policies.actorcritic
import go_ai.policies.value
from go_ai import data, game

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def matplot_format(state):
    """
    :param state:
    :return: A formatted state for matplot imshow.
    Only uses the piece and pass channels of the original state
    """
    assert len(state.shape) == 3
    return state.transpose(1, 2, 0)[:, :, [0, 1, 4]].astype(np.float)


def plot_move_distr(title, move_distr, valid_moves, scalar=None, pi=False):
    """

    :param title:
    :param move_distr:
    :param valid_moves:
    :param scalar:
    :return: A heat map of the move distribution including text displaying the range of the move values as well as an
    optional arbitrary scalar value of the whole move distribution
    """

    board_size = int((len(move_distr) - 1) ** 0.5)
    plt.axis('off')
    valid_values = np.extract(valid_moves[:-1] == 1, move_distr[:-1])
    invalid_moves = 1 - valid_moves
    pass_val = float(move_distr[-1])
    move_distr = np.ma.masked_array(move_distr, mask=invalid_moves)

    vmin = np.nanmin(valid_values) if len(valid_values) > 0 else 0
    vmax = np.nanmax(valid_values) if len(valid_values) > 0 else 0
    plt.title(title + (' ' if scalar is None else ' {:.3f}S'.format(scalar))
              + '\n{:.3f}L '.format(vmin)
              + '{:.3f}H '.format(vmax)
              + '{:.3f}P'.format(pass_val))

    if pi:
        vmin, vmax = 0, 1
    plt.imshow(np.reshape(move_distr[:-1], (board_size, board_size)), vmin=vmin, vmax=vmax)


def action_1d_to_2d(action_1d, board_width):
    """
    Converts 1D action to 2D or None if it's a pass
    """
    if action_1d == board_width ** 2:
        action = None
    else:
        action = (action_1d // board_width, action_1d % board_width)
    return action


def state_responses(policy: go_ai.policies.Policy, traj: game.Trajectory):
    states = np.array(traj.states)

    all_qs_list, state_vals = measure_vals(traj.actions, policy, states)
    all_qs = np.array(all_qs_list)
    layers_expanded = all_qs.shape[1]

    valid_moves = data.batch_valid_moves(states)

    n = len(traj)
    num_cols = 3 + layers_expanded

    fig = plt.figure(figsize=(num_cols * 2.5, n * 2))
    black_won = traj.get_winner()
    qs_title = 'Layer Qs' if isinstance(policy, go_ai.policies.value.Value) else 'Prior Pi'

    for i in tqdm(range(n), desc='Plots'):
        curr_col = 1

        plt.subplot(n, num_cols, curr_col + num_cols * i)
        plot_state(black_won, i, n, traj)
        curr_col += 1

        for j in range(layers_expanded):
            plt.subplot(n, num_cols, curr_col + num_cols * i)
            plot_move_distr(qs_title, all_qs[i, j], valid_moves[i], scalar=state_vals[i].item())
            curr_col += 1

        plt.subplot(n, num_cols, curr_col + num_cols * i)
        plot_move_distr('Model', traj.pis[i], valid_moves[i], pi=True)
        curr_col += 1

    plt.tight_layout()
    return fig


def plot_state(black_won, i, n, traj):
    terminal = int(i == n - 1)
    state = traj.states[i]
    board_size = state.shape[1]
    plt.axis('off')
    if i > 0:
        prev_action = action_1d_to_2d(traj.actions[i - 1], board_size)
        board_title = 'Action: {}\n'.format(prev_action)
        if i == n - 1:
            action_took = action_1d_to_2d(traj.actions[i], board_size)
            board_title += 'Action Taken: {}\n'.format(action_took)
    else:
        board_title = 'Initial Board\n'
    win = black_won if i % 2 == 0 else -black_won
    board_title += '{:.0f}R {}T, {}W'.format(traj.rewards[i], terminal, win)
    plt.title(board_title)
    plt.imshow(matplot_format(state))


def measure_vals(actions, policy, states):
    state_vals = []
    all_qs = []
    go_env = gym.make('gym_go:go-v0', size=states[0].shape[1])
    for step, (state, prev_action) in tqdm(enumerate(zip(states, actions)), desc='Heat Maps'):
        pi, qs, rootnode = policy(go_env, step=step, debug=True)
        if isinstance(policy, go_ai.policies.value.Value):
            state_val = policy.val_func(state[np.newaxis])
        else:
            _, state_val = policy.ac_func(state[np.newaxis])
        state_val = state_val[0]
        state_vals.append(state_val)
        all_qs.append(qs)
        go_env.step(prev_action)
    return all_qs, state_vals


def plot_traj_fig(go_env, policy: go_ai.policies.Policy, outpath):
    """
    Plays out a self-play game
    :param go_env:
    :param policy_args:
    :return: A plot of the game including the policy's responses to each state
    """
    go_env.reset()
    _, _, traj = game.pit(go_env, black_policy=policy, white_policy=policy)
    state_responses(policy, traj)
    plt.savefig(outpath)
    plt.close()


def plot_symmetries(next_state, outpath):
    symmetrical_next_states = GoGame.get_symmetries(next_state)

    cols = len(symmetrical_next_states)
    plt.figure(figsize=(3 * cols, 3))
    for i, state in enumerate(symmetrical_next_states):
        plt.subplot(1, cols, i + 1)
        plt.imshow(matplot_format(state))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def convert_to_secs(time_str):
    dur = time.strptime(time_str, '%H:%M:%S')
    secs = 3600 * dur.tm_hour + 60 * dur.tm_min + dur.tm_sec
    return secs


def convert_to_hours(time_str):
    return convert_to_secs(time_str) / 3600


def plot_stats(stats_path, outdir):
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
    plt.savefig(os.path.join(outdir, 'elos.pdf'))
    plt.close()

    # Win rate against random and greedy
    plt.figure()
    plt.title('Winrates against Baseline Models')
    plt.plot(df['HOURS'], df['R_WR'])
    plt.plot(df['HOURS'], df['G_WR'])
    plt.xlabel('Hours')
    plt.ylabel('Winrate')
    plt.legend(['Random', 'Greedy'])
    plt.savefig(os.path.join(outdir, 'winrates.pdf'))
    plt.close()

    # Loss and accuracy
    plt.figure()
    plt.title('Losses')
    plt.plot(df['HOURS'], df['C_LOSS'])
    plt.plot(df['HOURS'], df['A_LOSS'])
    plt.xlabel('Hours')
    plt.ylabel('Loss')
    plt.legend(['Critic', 'Actor'])
    plt.savefig(os.path.join(outdir, 'loss.pdf'))
    plt.close()

    plt.figure()
    plt.title('Accuracy')
    plt.plot(df['HOURS'], df['C_ACC'])
    plt.plot(df['HOURS'], df['A_ACC'])
    plt.xlabel('Hours')
    plt.ylabel('Accuracy')
    plt.legend(['Critic', 'Actor'])
    plt.savefig(os.path.join(outdir, 'acc.pdf'))
    plt.close()
