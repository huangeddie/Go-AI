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
    # input = layers.Input(shape=(board_size, board_size, 6), name="board")
    # x = layers.Conv2D(64, kernel_size=3, padding="same")(input)
    #
    # dense_block = add_dense_block(x, num_layers=8, growth_rate=12)
    #
    # out = add_head(dense_block)
    #
    # model = tf.keras.Model(input, out, name='dense_valnet')

    model = tf.keras.Sequential([
        layers.Input(shape=(board_size, board_size, 6), name="board"),
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(256),
        layers.ReLU(),
        layers.Dense(1, activation='tanh'),
    ])

    model.compile(optimizer='adam', loss='mse')

    return model


def make_val_func(val_net):
    def forward_func(states):
        return val_net(states.transpose(0, 2, 3, 1).astype(np.float32), training=False).numpy()

    return forward_func


def optimize_val_net(value_model_args: policies.PolicyArgs, replay_data, batch_size):
    """
    Loads in parameters from disk and updates them from the batched memory (saves the new parameters back to disk)
    :param value_model_args:
    :param actor_critic:
    :param just_critic: Dictates whether we update just the critic portion
    :param replay_data:
    :param optimizer:
    :param iteration:
    :param tb_metrics:
    :return:
    """
    # Load model from disk
    model = tf.keras.models.load_model(value_model_args.model_path)

    # Iterate through data
    model.fit(replay_data[0].transpose(0, 2, 3, 1), replay_data[5], epochs=1, batch_size=batch_size)

    # Update the weights on disk
    model.save(value_model_args.model_path)


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
