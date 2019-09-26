import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from go_ai import policies
from go_ai import data
from tqdm import tqdm

GoGame = gym.make('gym_go:go-v0', size=0).gogame

def add_dense_layer(input, growth_rate):
    x = layers.BatchNormalization()(input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4 * growth_rate, kernel_size=1)(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(growth_rate, kernel_size=3, padding='same')(x)

    return x

def add_dense_block(input, num_layers, growth_rate):
    x = input

    for l in range(num_layers):
        y = add_dense_layer(x, growth_rate)
        x = layers.Concatenate()([y, x])

    return x

def add_head(input):
    x = layers.Conv2D(1, kernel_size=1)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)

    x = layers.Dense(1, activation='tanh')(x)
    return x


def make_val_net(board_size):
    input = layers.Input(shape=(board_size, board_size, 6), name="board")
    x = layers.Conv2D(64, kernel_size=3, padding="same")(input)

    dense_block = add_dense_block(x, num_layers=8, growth_rate=12)

    out = add_head(dense_block)

    model = tf.keras.Model(input, out, name='dense val net')

    return model


def make_val_func(val_net):
    def forward_func(states):
        return val_net(states.transpose(0, 2, 3, 1).astype(np.float32), training=False).numpy()

    return forward_func


def optimize_val_net(value_model_args: policies.PolicyArgs, batched_mem, learning_rate):
    """
    Loads in parameters from disk and updates them from the batched memory (saves the new parameters back to disk)
    :param value_model_args:
    :param actor_critic:
    :param just_critic: Dictates whether we update just the critic portion
    :param batched_mem:
    :param optimizer:
    :param iteration:
    :param tb_metrics:
    :return:
    """
    # Load model from disk
    model = tf.keras.models.load_model(value_model_args.model_path)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Define criterion
    mean_squared_error = tf.keras.losses.MeanSquaredError()

    # Metrics
    loss_metric = tf.keras.metrics.Mean()
    pred_metric = tf.keras.metrics.Accuracy()

    # Iterate through data
    pbar = tqdm(batched_mem, desc='Updating', leave=True, position=0)
    for states, actions, next_states, rewards, terminals, wins in pbar:
        wins = wins[:, np.newaxis]

        # Augment states
        states = data.batch_random_symmetries(states)

        with tf.GradientTape() as tape:
            state_vals = model(states.transpose(0, 2, 3, 1).astype(np.float32), training=True)

            # Critic
            assert state_vals.shape == wins.shape
            val_loss = mean_squared_error(wins, state_vals)
            loss_metric.update_state(val_loss)
            wins_01 = (np.copy(wins) + 1) / 2
            pred_metric.update_state(wins_01, state_vals > 0)

        # compute and apply gradients
        gradients = tape.gradient(val_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Metrics
        pbar.set_postfix_str('{:.1f}% ACC, {:.3f}VL'.format(100 * pred_metric.result().numpy(),
                                                            loss_metric.result().numpy()))
    # Update the weights on disk
    model.save_weights(value_model_args.model_path)


def greedy_val_func(states):
    board_area = GoGame.get_action_size(states[0]) - 1

    vals = []
    for state in states:
        black_area, white_area = GoGame.get_areas(state)
        if GoGame.get_game_ended(state):
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
