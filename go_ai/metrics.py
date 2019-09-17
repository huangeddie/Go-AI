import io
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

import go_ai.models
from go_ai import data, mcts


def plot_move_distr(title, move_distr, valid_moves, scalar=None):
    """
    Takes in a 1d array of move values and plots its heatmap
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


def state_responses_helper(actor_critic, states, taken_actions, next_states, rewards, terminals, wins, mcts_move_probs):
    """
    Helper function for state_responses
    :param actor_critic:
    :param states:
    :param taken_actions:
    :param next_states:
    :param rewards:
    :param terminals:
    :param wins:
    :param mcts_move_probs:
    :return:
    """

    def action_1d_to_2d(action_1d, board_width):
        """
        Converts 1D action to 2D or None if it's a pass
        """
        if action_1d == board_width ** 2:
            action = None
        else:
            action = (action_1d // board_width, action_1d % board_width)
        return action

    board_size = states[0].shape[0]

    move_probs, move_vals = go_ai.models.forward_pass(states, actor_critic, training=False)
    state_vals = tf.reduce_sum(move_probs * move_vals, axis=1)

    valid_moves = go_ai.models.get_valid_moves(states)

    num_states = states.shape[0]
    num_cols = 4

    fig = plt.figure(figsize=(num_cols * 2.5, num_states * 2))
    for i in range(num_states):
        curr_col = 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plt.axis('off')
        plt.title('Board')
        plt.imshow(states[i][:, :, [0, 1, 4]].astype(np.float))
        curr_col += 1

        if mcts_move_probs is None:
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr('Critic', move_vals[i], valid_moves[i],
                            scalar=state_vals[i].numpy())
        else:
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            plot_move_distr('MCTS', mcts_move_probs[i], valid_moves[i])
        curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plot_move_distr('Actor{}'.format(' Critic' if mcts_move_probs is not None else ''),
                        move_probs[i], valid_moves[i],
                        scalar=state_vals[i].numpy().item())
        curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        plt.axis('off')
        plt.title('Taken Action: {}\n{:.0f}R {}T, {}W'
                  .format(action_1d_to_2d(taken_actions[i], board_size), rewards[i], terminals[i], wins[i]))
        plt.imshow(next_states[i][:, :, [0, 1, 4]].astype(np.float))
        curr_col += 1

    plt.tight_layout()
    return fig


def state_responses(actor_critic, replay_mem):
    """
    :param actor_critic: The model
    :param replay_mem: List of events
    :return: The figure visualizing responses of the model
    on those events
    """
    states, actions, next_states, rewards, terminals, wins, mc_pis = data.replay_mem_to_numpy(replay_mem)
    assert len(states[0].shape) == 3 and states[0].shape[0] == states[0].shape[1], states[0].shape

    fig = state_responses_helper(actor_critic, states, actions, next_states, rewards, terminals, wins, mc_pis)
    return fig


def gen_traj_fig(go_env, actor_critic, temp_func, max_steps, mc_sims):
    traj, _ = data.self_play(go_env, policy=actor_critic, max_steps=max_steps, mc_sims=mc_sims,
                             temp_func=temp_func, get_symmetries=False)
    fig = state_responses(actor_critic, traj)
    return fig


def plot_symmetries(go_env, actor_critic, outpath):
    mct_forward = go_ai.models.make_mcts_forward(actor_critic)

    mem = []
    state = go_env.reset()
    action = (1, 2)
    action_1d = go_env.action_2d_to_1d(action)
    next_state, reward, done, info = go_env.step(action)
    mct = mcts.MCTree(state, mct_forward)
    mc_pi = mct.get_action_probs(max_num_searches=0, temp=1)
    data.add_to_replay_mem(mem, state, action_1d, next_state, reward, done, 0, mc_pi)

    fig = state_responses(actor_critic, mem)
    fig.savefig(outpath)
    plt.close()


def figure_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def log_to_tensorboard(summary_writer, metrics, step, go_env, actor_critic, figpath=None):
    """
    Logs metrics to tensorboard.
    Also resets keras metrics after use
    """
    with summary_writer.as_default():
        # Keras metrics
        for key, metric in metrics.items():
            tf.summary.scalar(key, metric.result(), step=step)

        reset_metrics(metrics)

        # Plot samples of states and response heatmaps
        board_size = go_env.size
        fig = gen_traj_fig(go_env, actor_critic, lambda x: 1 / 64, 2 * board_size ** 2, 0)
        if figpath is not None:
            fig.savefig(figpath)
        tf.summary.image("Trajectory and Responses", figure_to_image(fig), step=step)


def reset_metrics(metrics):
    for key, metric in metrics.items():
        metric.reset_states()


def evaluate(go_env, policy, opponent, max_steps, num_games, mc_sims, temp_func):
    win_metric = tf.keras.metrics.Mean()

    pbar = tqdm(range(num_games), desc='Evaluation', leave=False, position=0)
    for episode in pbar:
        if episode % 2 == 0:
            black_won = data.pit(go_env, policy, opponent, max_steps, mc_sims, temp_func)
            win = (black_won + 1) / 2

        else:
            black_won = data.pit(go_env, opponent, policy, max_steps, mc_sims, temp_func)
            win = (-black_won + 1) / 2

        win_metric.update_state(win)
        pbar.set_postfix_str('{} {:.1f}%'.format(win, 100 * win_metric.result().numpy()))

    return win_metric.result().numpy()
