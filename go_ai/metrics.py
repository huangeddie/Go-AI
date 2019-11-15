import gym
import numpy as np
from matplotlib import pyplot as plt

import go_ai.game
from go_ai import data, montecarlo, policies
from go_ai.montecarlo import tree
import queue
from tqdm import tqdm

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def matplot_format(state):
    """
    :param state:
    :return: A formatted state for matplot imshow.
    Only uses the piece and pass channels of the original state
    """
    assert len(state.shape) == 3
    return state.transpose(1, 2, 0)[:, :, [0, 1, 4]].astype(np.float)


def plot_move_distr(title, move_distr, valid_moves, vmin, vmax ,scalar=None):
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
    plt.title(title + (' ' if scalar is None else ' {:.3f}S'.format(scalar))
              + '\n{:.3f}L '.format(np.min(valid_values) if len(valid_values) > 0 else 0)
              + '{:.3f}H '.format(np.max(valid_values) if len(valid_values) > 0 else 0)
              + '{:.3f}P'.format(pass_val))
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

    move_probs = []
    state_vals = []
    qvals = []
    for step, (state, prev_action) in tqdm(enumerate(zip(states, taken_actions)), desc='Heat Maps'):
        if step == 0:
            policy.reset(state)
        pi = policy(state, step)
        move_probs.append(pi)
        policy.step(prev_action)

        if isinstance(policy, policies.MCTS):
            state_val = policy.val_func(state[np.newaxis])[0]
            qs, _ = montecarlo.qs_from_stateval(state[np.newaxis], policy.val_func)
            qs = qs[0]

            state_vals.append(state_val)
            qvals.append(qs)

    valid_moves = data.batch_valid_moves(states)

    num_states = states.shape[0]
    num_cols = 3 if isinstance(policy, policies.MCTS) else 2

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

        if isinstance(policy, policies.MCTS):
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr('Q Vals', qvals[i], valid_moves[i], vmin=-1, vmax=1, scalar=state_vals[i].item())
            curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plot_move_distr('Model', move_probs[i], valid_moves[i], vmin=0, vmax=1)
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


def plot_mct(tree: tree.MCTree, outpath, max_layers=8, max_branch=8):
    """
    :param root_node: The Node to start plotting from
    :param max_layers: The number of layers to plot (1st layer is the root)
    :param max_branch: The max number of canon_children show per node
    :return: A plot with each layer of the MCT states in a row
    """
    max_width = max_branch ** (max_layers - 1)
    grid = np.empty((max_layers, max_width), dtype=object)

    # Traverse tree to flatten into columns
    # Consists of (node, level) pairs
    root_node = tree.root

    que = queue.SimpleQueue()
    que.put((root_node, 0))
    curr_x = 0
    curr_y = -1
    while not que.empty():
        node, level = que.get()
        assert node is not None
        # If we are not moving down in the grid, move right
        if level <= curr_y:
            curr_x += 1
        else:
            curr_x = 0
        curr_y = level
        grid[curr_y, curr_x] = node
        if level < max_layers - 1 and not node.is_leaf():
            canon_children = list(filter(lambda child: child is not None, node.canon_children))
            # Sort in ascending order so most visited goes on top of stack
            canon_children = sorted(canon_children, key=lambda c: np.sum(c.move_visits), reverse=True)
            if max_branch:
                # Take last k canon_children
                canon_children = canon_children[:max_branch]

            for c in canon_children:
                que.put((c, curr_y + 1))

    # Trim empty columns from grid
    grid = grid[:, :curr_x + 1]

    plt.figure(figsize=(grid.shape[1] * 2, grid.shape[0] * 2))
    # Qvals
    root_qs = tree.root.latest_qs()
    plt.subplot(grid.shape[0], grid.shape[1], 2)
    plt.title('Q Vals')
    plt.bar(np.arange(len(root_qs)), root_qs)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            node = grid[i, j]
            if node is None:
                continue

            if node.actiontook is not None:
                action = action_1d_to_2d(node.actiontook, node.state.shape[1])
                qval = node.parent.latest_q(node.actiontook)
            else:
                assert node.parent is None
                action = None
                qval = 0
            visits = node.visits
            value = node.latest_value()
            prior_val = node.prior_value

            plt.subplot(grid.shape[0], grid.shape[1], grid.shape[1] * i + j + 1)
            plt.axis('off')
            plt.title(f'{visits}N\n{prior_val:.2f}PV {value:.2f}V\n{action}A {qval:.2f}Q')
            plt.imshow(matplot_format(node.state))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
