import gym
import numpy as np
from matplotlib import pyplot as plt

import go_ai.game
from go_ai import data, montecarlo, policies

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def matplot_format(state):
    """
    :param state:
    :return: A formatted state for matplot imshow.
    Only uses the piece and pass channels of the original state
    """
    assert len(state.shape) == 3
    return state.transpose(1, 2, 0)[:, :, [0, 1, 4]].astype(np.float)


def plot_move_distr(title, move_distr, valid_moves, scalar=None):
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
    assert np.isnan(move_distr).any() == False, move_distr
    pass_val = float(move_distr[-1])
    plt.title(title + (' ' if scalar is None else ' {:.3f}S').format(scalar)
              + '\n{:.3f}L {:.3f}H {:.3f}P'.format(np.min(valid_values)
                                                   if len(valid_values) > 0 else 0,
                                                   np.max(valid_values)
                                                   if len(valid_values) > 0 else 0,
                                                   pass_val))
    plt.imshow(np.reshape(move_distr[:-1], (board_size, board_size)))

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

    move_probs = []
    for step, state in enumerate(states):
        pi = policy(state, step)
        move_probs.append(pi)

    state_vals = None
    qvals = None
    if isinstance(policy, policies.QTempPolicy):
        state_vals = policy.val_func(states)
        qvals, _ = montecarlo.qval_from_stateval(states, policy.val_func)

    valid_moves = data.batch_valid_moves(states)

    num_states = states.shape[0]
    num_cols = 4 if isinstance(policy, policies.QTempPolicy) else 3

    fig = plt.figure(figsize=(num_cols * 2.5, num_states * 2))
    for i in range(num_states):
        curr_col = 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plt.axis('off')
        plt.title('Board')
        plt.imshow(matplot_format(states[i]))
        curr_col += 1

        if isinstance(policy, policies.QTempPolicy):
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr('Q Vals', qvals[i], valid_moves[i], scalar=state_vals[i].item())
            curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plot_move_distr('Model', move_probs[i], valid_moves[i])
        curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plt.axis('off')
        plt.title('Taken Action: {}\n{:.0f}R {}T, {}W'
                  .format(action_1d_to_2d(taken_actions[i], board_size), rewards[i], terminals[i], wins[i]))
        plt.imshow(matplot_format(next_states[i]))
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


def gen_traj_fig(go_env, policy: policies.Policy, outpath):
    """
    Plays out a self-play game
    :param go_env:
    :param policy_args:
    :return: A plot of the game including the policy's responses to each state
    """
    go_env.reset()
    _, traj = go_ai.game.pit(go_env, black_policy=policy, white_policy=policy, get_traj=True)
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


def plot_mct(root_node, max_layers, max_branch=None):
    """
    :param root_node: The Node to start plotting from
    :param max_layers: The number of layers to plot (1st layer is the root)
    :param max_branch: The max number of children show per node
    :return: A plot with each layer of the MCT states in a row
    """
    max_width = max_branch ** (max_layers - 1)
    grid = np.empty((max_layers, max_width), dtype=object)

    # Traverse tree to flatten into columns
    # Consists of (node, level) pairs
    stack = [(root_node, 0)]
    curr_x = 0
    curr_y = -1
    while stack:
        node, level = stack.pop()
        # If we are not moving down in the grid, move right
        if level <= curr_y:
            curr_x += 1
        curr_y = level
        grid[curr_y, curr_x] = node
        if level < max_layers - 1 and not node.is_leaf():
            children = list(filter(
                lambda child: child is not None and child.visited(), node.children))
            # Sort in ascending order so most visited goes on top of stack
            children = sorted(children, key=lambda c: c.N)
            if max_branch:
                # Take last k children
                children = children[-max_branch:]
            pairs = [(c, curr_y + 1) for c in children]
            stack.extend(pairs)

    # Trim empty columns from grid
    grid = grid[:, :curr_x + 1]

    fig = plt.figure(figsize=(grid.shape[1] * 2, grid.shape[0] * 2))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            node = grid[i, j]
            if node is None:
                continue

            if node.last_action:
                action = action_1d_to_2d(node.last_action, node.state.shape[1])
            else:
                action = None
            visits = node.N
            value = node.V

            plt.subplot(grid.shape[0], grid.shape[1], grid.shape[1] * i + j + 1)
            plt.axis('off')
            plt.title('A={} N={} V={:.2f}'.format(action, visits, value))
            plt.imshow(matplot_format(node.state))
    plt.tight_layout()
    return fig


def gen_mct_plot(go_env, policy_args, max_layers, max_branch=None):
    """
    Displays a MCT after a self-play game
    :param go_env:
    :param policy_args: Arguments for a monte_carlo policy
    :return: A plot of the MCT
    """
    assert policy_args.mode == 'monte_carlo'
    policy = pmaker.make_policy(policy_args)
    policy.tree.save_root()

    go_env.reset()
    data.pit(go_env, black_policy=policy, white_policy=policy)

    root = policy.tree.orig_root
    fig = plot_mct(root, max_layers, max_branch)
    return fig