import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from go_ai import data, policies, montecarlo
from sklearn.preprocessing import normalize


def make_model(board_size):
    action_size = board_size ** 2 + 1

    inputs = layers.Input(shape=(board_size, board_size, 6), name="state")
    invalid_values = layers.Input(shape=(action_size,), name="invalid_values")

    # Actor
    move_probs = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    move_probs = layers.BatchNormalization()(move_probs)
    move_probs = layers.ReLU()(move_probs)
    move_probs = layers.Conv2D(64, kernel_size=3, padding='same')(move_probs)
    move_probs = layers.BatchNormalization()(move_probs)
    move_probs = layers.ReLU()(move_probs)

    move_probs = layers.Conv2D(2, kernel_size=1)(move_probs)
    move_probs = layers.BatchNormalization()(move_probs)
    move_probs = layers.ReLU()(move_probs)
    move_probs = layers.Flatten()(move_probs)
    move_probs = layers.Dense(action_size)(move_probs)
    move_probs = layers.Add()([move_probs, invalid_values])
    move_probs = layers.Softmax(name="move_probs")(move_probs)

    # Critic
    move_vals = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    move_vals = layers.BatchNormalization()(move_vals)
    move_vals = layers.ReLU()(move_vals)
    move_vals = layers.Conv2D(64, kernel_size=3, padding='same')(move_vals)
    move_vals = layers.BatchNormalization()(move_vals)
    move_vals = layers.ReLU()(move_vals)

    move_vals = layers.Conv2D(1, kernel_size=1)(move_vals)
    move_vals = layers.BatchNormalization()(move_vals)
    move_vals = layers.ReLU()(move_vals)
    move_vals = layers.Flatten()(move_vals)

    move_vals = layers.Dense(action_size)(move_vals)
    move_vals = layers.ReLU()(move_vals)
    move_vals = layers.Dense(1, activation='tanh', name="value")(move_vals)

    model = tf.keras.Model(inputs=[inputs, invalid_values], outputs=[move_probs, move_vals],
                           name='actor_critic')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss={
            'move_probs': tf.keras.losses.CategoricalCrossentropy(),
            'value': tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            'move_probs': [tf.keras.metrics.CategoricalAccuracy()],
        },
        loss_weight={'move_probs': 1, 'value': 1}
    )

    return model


def forward_pass(states, network, training):
    """
    Since the neural nets take in more than one parameter,
    this functions serves as a wrapper to forward pass the data through the networks.
    :param states:
    :param network:
    :param training: Boolean parameter for layers like BatchNorm
    :return: action probs and state vals
    """
    invalid_values = data.batch_invalid_values(states)
    return network([states.transpose(0, 2, 3, 1).astype(np.float32),
                    invalid_values.astype(np.float32)], training=training)


def make_forward_func(network):
    """
    :param network:
    :return: A more simplified forward pass function that just takes in states and outputs the action probs and
    state values in numpy form
    """

    def forward_func(states):
        action_probs, state_vals = forward_pass(states, network, training=False)
        return action_probs.numpy(), state_vals.numpy()

    return forward_func


def optimize(policy_args: policies.PolicyArgs, replay_data, batch_size):
    """
    Loads in parameters from disk and updates them from the batched memory (saves the new parameters back to disk)
    :param actor_critic:
    :param just_critic: Dictates whether we update just the critic portion
    :param replay_data:
    :param optimizer:
    :param iteration:
    :param tb_metrics:
    :return:
    """
    # Load model from disk
    model = tf.keras.models.load_model(policy_args.model_path)
    forward_func = make_forward_func(model)

    # Augment states
    states = replay_data[0]
    states = data.batch_random_symmetries(states)
    invalid_values = data.batch_invalid_values(states)

    # Policy Evaluation
    data_size = states.shape[0]
    board_size = policy_args.board_size
    action_size = board_size ** 2 + 1

    # Wins
    wins = replay_data[5]

    # Fit
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss={
            'move_probs': tf.keras.losses.CategoricalCrossentropy(),
            'value': tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            'move_probs': [tf.keras.metrics.CategoricalAccuracy()],
        },
        loss_weight={'move_probs': 0, 'value': 1}
    )
    fake_pi = np.zeros((data_size, action_size))
    fake_pi[:, -1] = 1
    model.fit({'state': states.transpose(0, 2, 3, 1), 'invalid_values': invalid_values},
              {'value': wins, 'move_probs': fake_pi},
              batch_size=batch_size, epochs=1)

    # Policy Iteration

    # Get Q values and then target pis of current critic
    _, qvals, _ = montecarlo.piqval_from_actorcritic(states, forward_func)
    valid_moves = data.batch_valid_moves(states)
    min_qvals = np.min(qvals, axis=1, keepdims=True)
    qvals -= min_qvals
    qvals += 1e-7 * valid_moves
    qvals *= valid_moves

    max_qs = np.argmax(qvals, axis=1)
    target_pis = np.eye(action_size)[max_qs]

    assert (np.sum(target_pis, axis=1) == 1).all()
    assert (target_pis[np.where(invalid_values != 0)] == 0).all()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss={
            'move_probs': tf.keras.losses.CategoricalCrossentropy(),
            'value': tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            'move_probs': [tf.keras.metrics.CategoricalAccuracy()],
        },
        loss_weight={'move_probs': 1, 'value': 1}
    )
    model.fit({'state': states.transpose(0, 2, 3, 1), 'invalid_values': invalid_values},
              {'move_probs': target_pis, 'value': wins},
              batch_size=batch_size, epochs=1)

    # Update the weights on disk
    model.save(policy_args.model_path)
