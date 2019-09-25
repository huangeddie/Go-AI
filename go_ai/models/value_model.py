import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from go_ai import data
from tqdm import tqdm

go_env = gym.make('gym_go:go-v0', size=0)
govars = go_env.govars
gogame = go_env.gogame


def make_val_net(board_size):
    model = tf.keras.Sequential([
        layers.Input(shape=(board_size, board_size, 6), name="board"),

        layers.Conv2D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(64, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(8, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Flatten(),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Dense(1, activation='tanh', name="value")
    ])

    return model


def make_forward_func(val_net):
    def forward_func(states):
        return val_net(states.transpose(0, 2, 3, 1).astype(np.float32), training=False).numpy()

    return forward_func


def optimize_val_net(value_model_args, batched_mem, learning_rate, tb_metrics):
    """
    Loads in parameters from disk and updates them from the batched memory (saves the new parameters back to disk)
    :param actor_critic:
    :param just_critic: Dictates whether we update just the critic portion
    :param batched_mem:
    :param optimizer:
    :param iteration:
    :param tb_metrics:
    :return:
    """
    # Load model from disk
    val_net = make_val_net(value_model_args['board_size'])
    val_net.load_weights(value_model_args['model_path'])
    forward_func = make_forward_func(val_net)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Define criterion
    mean_squared_error = tf.keras.losses.MeanSquaredError()

    # Iterate through data
    pbar = tqdm(batched_mem, desc='Updating', leave=True, position=0)
    for states, actions, next_states, rewards, terminals, wins in pbar:
        batch_size = states.shape[0]
        wins = wins[:, np.newaxis]

        # Augment states
        states = data.random_symmetries(states)

        with tf.GradientTape() as tape:
            state_vals = val_net(states.transpose(0, 2, 3, 1).astype(np.float32), training=True)

            # Critic
            assert state_vals.shape == wins.shape
            val_loss = mean_squared_error(wins, state_vals)
            tb_metrics['val_loss'].update_state(val_loss)
            wins_01 = (np.copy(wins) + 1) / 2
            tb_metrics['pred_win_acc'].update_state(wins_01, state_vals > 0)

        # compute and apply gradients
        gradients = tape.gradient(val_loss, val_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, val_net.trainable_variables))

        # Metrics
        pbar.set_postfix_str('{:.1f}% ACC, {:.3f}VL'.format(100 * tb_metrics['pred_win_acc'].result().numpy(),
                                                            tb_metrics['val_loss'].result().numpy()))
    # Update the weights on disk
    val_net.save_weights(value_model_args['model_path'])


def greedy_vals(states):
    board_area = gogame.get_action_size(states[0]) - 1

    vals = []
    for state in states:
        black_area, white_area = gogame.get_areas(state)
        if gogame.get_game_ended(state):
            if black_area > white_area:
                val = 1
            elif black_area < white_area:
                val = -1
            else:
                val = 0
        else:
            val = (black_area - white_area) / board_area
        vals.append(val)
    vals = np.array(vals, dtype=np.float)
    return vals[:, np.newaxis]
