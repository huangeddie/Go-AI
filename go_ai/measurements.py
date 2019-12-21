import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import go_ai.game
from go_ai import data, policies

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
    move_distr[np.where(invalid_moves)] = np.nan
    pass_val = float(move_distr[-1])

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


def state_responses_helper(policy: policies.Policy, states, taken_actions, next_states, rewards, terminals, wins):
    """
    Helper function for state_responses
    :param policy_args:
    :param states:
    :param taken_actions:
    :param next_states:
    :param rewards:
    :param terminals:
    :param wins:
    :return:
    """
    board_size = states[0].shape[1]

    all_pi = []
    state_vals = []
    all_prior_qs = []
    all_post_qs = []
    go_env = gym.make('gym_go:go-v0', size=states[0].shape[1])
    for step, (state, prev_action) in tqdm(enumerate(zip(states, taken_actions)), desc='Heat Maps'):
        if isinstance(policy, policies.Value) or isinstance(policy, policies.ActorCritic):
            pi, prior_qs, post_qs = policy(go_env, step=step, get_qs=True)
            state_val = policy.val_func(state[np.newaxis])[0]

            state_vals.append(state_val)
            all_prior_qs.append(prior_qs)
            all_post_qs.append(post_qs)
        else:
            pi = policy(go_env, step=step)
        all_pi.append(pi)
        go_env.step(prev_action)

    valid_moves = data.batch_valid_moves(states)

    num_states = states.shape[0]
    if isinstance(policy, policies.Value) or isinstance(policy, policies.ActorCritic):
        num_cols = 4
    else:
        num_cols = 2

    fig = plt.figure(figsize=(num_cols * 2.5, num_states * 2))
    for i in tqdm(range(num_states), desc='Plots'):
        curr_col = 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plt.axis('off')
        if i > 0:
            prev_action = action_1d_to_2d(taken_actions[i - 1], board_size)
            board_title = 'Action: {}\n'.format(prev_action)
            if i == num_states - 1:
                action_took = action_1d_to_2d(taken_actions[i], board_size)
                board_title += 'Action Taken: {}\n'.format(action_took)
        else:
            board_title = 'Initial Board\n'
        board_title += '{:.0f}R {}T, {}W'.format(rewards[i], terminals[i], wins[i])

        plt.title(board_title)
        plt.imshow(matplot_format(states[i]))
        curr_col += 1

        if isinstance(policy, policies.Value) or isinstance(policy, policies.ActorCritic):
            prior_title = 'Prior Qs' if isinstance(policy, policies.Value) else 'Prior Pi'
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr(prior_title, all_prior_qs[i], valid_moves[i], scalar=state_vals[i].item())
            curr_col += 1

            post_title = 'Post Qs' if isinstance(policy, policies.Value) else 'Visits'
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr(post_title, all_post_qs[i], valid_moves[i], scalar=state_vals[i].item())
            curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plot_move_distr('Model', all_pi[i], valid_moves[i], pi=True)
        curr_col += 1

    plt.tight_layout()
    return fig


def state_responses(policy: policies.Policy, replay_mem):
    """
    :param policy_args: The model
    :param replay_mem: List of events
    :return: The figure visualizing responses of the model
    on those events
    """
    states, actions, next_states, rewards, terminals, wins = data.replaylist_to_numpy(replay_mem)
    assert len(states[0].shape) == 3 and states[0].shape[1] == states[0].shape[2], states[0].shape

    fig = state_responses_helper(policy, states, actions, next_states, rewards, terminals, wins)
    return fig


def plot_traj_fig(go_env, policy: policies.Policy, outpath):
    """
    Plays out a self-play game
    :param go_env:
    :param policy_args:
    :return: A plot of the game including the policy's responses to each state
    """
    go_env.reset()
    _, _, traj = go_ai.game.pit(go_env, black_policy=policy, white_policy=policy, get_traj=True)
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
