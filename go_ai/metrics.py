import io
import logging

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from go_ai import rl_utils

def state_responses(actor_critic, states, taken_actions, next_states, rewards, terminals, wins, mcts_move_probs):
    def action_1d_to_2d(action_1d, board_width):
        """
        Converts 1D action to 2D or None if it's a pass
        """
        if action_1d == board_width ** 2:
            action = None
        else:
            action = (action_1d // board_width, action_1d % board_width)
        return action

    """
    Returns a figure of plots on the states and the models responses on those states
    """
    board_size = states[0].shape[0]

    move_probs, move_vals = rl_utils.forward_pass(states, actor_critic, training=False)
    state_vals = tf.reduce_sum(move_probs * move_vals, axis=1)

    valid_moves = rl_utils.get_valid_moves(states)

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


def sample_heatmaps(actor_critic, replay_mem, num_samples=3):
    states, actions, next_states, rewards, terminals, wins, mc_pis = rl_utils.replay_mem_to_numpy(replay_mem[:num_samples])
    assert len(states[0].shape) == 3 and states[0].shape[0] == states[0].shape[1], states[0].shape

    # Add latest terminal state
    got_terminal = False
    got_last_state = False
    for (state, action, next_state, reward, terminal, win, mc_pi) in reversed(replay_mem):
        add_obs = False
        if terminal and not got_terminal:
            got_terminal = True
            add_obs = True

        if np.sum(state[:2]) == 0 and not got_last_state:
            got_last_state = True
            add_obs = True

        if add_obs:
            states = np.append(states, state.transpose(1, 2, 0)[np.newaxis], axis=0)
            actions = np.append(actions, action)
            next_states = np.append(next_states, next_state.transpose(1, 2, 0)[np.newaxis], axis=0)
            rewards = np.append(rewards, reward)
            terminals = np.append(terminals, terminal)
            wins = np.append(wins, win)
            mc_pis = np.append(mc_pis, mc_pi[np.newaxis], axis=0)

        if got_terminal and got_last_state:
            break

    fig = state_responses(actor_critic, states, actions, next_states, rewards, terminals, wins, mc_pis)
    return fig


def log_to_tensorboard(summary_writer, metrics, step, replay_mem, actor_critic):
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
        logging.debug("Sampling heatmaps...")
        fig = sample_heatmaps(actor_critic, replay_mem, num_samples=2)
        tf.summary.image("model heat maps", plot_to_image(fig), step=step)


def plot_move_distr(title, move_distr, valid_moves, scalar=None):
    """
    Takes in a 1d array of move values and plots its heatmap
    """
    board_size = int((len(move_distr) - 1) ** 0.5)
    plt.axis('off')
    valid_values = np.extract(valid_moves[:-1] == 1, move_distr[:-1])
    assert np.isnan(move_distr).any() == False, move_distr
    pass_val = move_distr[-1] if isinstance(move_distr[-1], np.float) else move_distr[-1].numpy()
    plt.title(title + (' ' if scalar is None else ' {:.3f}S').format(scalar)
              + '\n{:.3f}L {:.3f}H {:.3f}P'.format(np.min(valid_values)
                                                   if len(valid_values) > 0 else 0,
                                                   np.max(valid_values)
                                                   if len(valid_values) > 0 else 0,
                                                   pass_val))
    plt.imshow(np.reshape(move_distr[:-1], (board_size, board_size)))


def plot_to_image(figure):
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


def reset_metrics(metrics):
    for key, metric in metrics.items():
        metric.reset_states()


def evaluate(go_env, policy, opponent, max_steps, num_games, mc_sims):
    avg_metric = tf.keras.metrics.Mean()

    pbar = tqdm_notebook(range(num_games), desc='Evaluating against former self', leave=False)
    for episode in pbar:
        if episode % 2 == 0:
            black_won = rl_utils.pit(go_env, policy, opponent, max_steps, mc_sims)
            avg_metric((black_won + 1) / 2)
        else:
            black_won = rl_utils.pit(go_env, opponent, policy, max_steps, mc_sims)
            avg_metric((-black_won + 1) / 2)
        pbar.set_postfix_str('{:.1f}%'.format(100 * avg_metric.result()))

    return avg_metric.result()