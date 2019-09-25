import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import go_ai.montecarlo
from go_ai import data
from go_ai.montecarlo import tree
from sklearn.preprocessing import normalize


def make_actor_critic(board_size, critic_mode='val_net', critic_activation='tanh'):
    action_size = board_size ** 2 + 1

    inputs = layers.Input(shape=(board_size, board_size, 6), name="board")
    valid_inputs = layers.Input(shape=(action_size,), name="valid_moves")
    invalid_values = layers.Input(shape=(action_size,), name="invalid_values")

    x = inputs

    x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Actor
    move_probs = layers.Conv2D(2, kernel_size=1)(x)
    move_probs = layers.BatchNormalization()(move_probs)
    move_probs = layers.ReLU()(move_probs)
    move_probs = layers.Flatten()(move_probs)
    move_probs = layers.Dense(action_size)(move_probs)
    move_probs = layers.Add()([move_probs, invalid_values])
    move_probs = layers.Softmax(name="move_probs")(move_probs)
    actor_out = move_probs

    # Critic
    move_vals = layers.Conv2D(1, kernel_size=1)(x)
    move_vals = layers.BatchNormalization()(move_vals)
    move_vals = layers.ReLU()(move_vals)
    move_vals = layers.Flatten()(move_vals)

    move_vals = layers.Dense(action_size)(move_vals)
    move_vals = layers.ReLU()(move_vals)
    if critic_mode == 'q_net':
        move_vals = layers.Dense(action_size, activation=critic_activation)(move_vals)
        move_vals = layers.Multiply(name="move_vals")([move_vals, valid_inputs])
    elif critic_mode == 'val_net':
        move_vals = layers.Dense(1, activation=critic_activation, name="value")(move_vals)
    else:
        raise Exception("Unknown critic mode")

    model = tf.keras.Model(inputs=[inputs, valid_inputs, invalid_values], outputs=[actor_out, move_vals],
                           name='actor_critic')
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
    invalid_moves = data.batch_invalid_moves(states)
    invalid_values = data.batch_invalid_values(states)
    valid_moves = 1 - invalid_moves
    return network([states.transpose(0, 2, 3, 1).astype(np.float32),
                    valid_moves.astype(np.float32),
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


def optimize_actor_critic(policy_args, batched_mem, learning_rate, tb_metrics):
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
    actor_critic = make_actor_critic(policy_args['board_size'])
    actor_critic.load_weights(policy_args['model_path'])
    forward_func = make_forward_func(actor_critic)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Define criterion
    cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    mean_squared_error = tf.keras.losses.MeanSquaredError()

    # Iterate through data
    pbar = tqdm(batched_mem, desc='Updating', leave=True, position=0)
    for states, actions, next_states, rewards, terminals, wins in pbar:
        batch_size = states.shape[0]
        wins = wins[:, np.newaxis]

        # Augment states
        states = data.batch_random_symmetries(states)

        # Get Q values of current critic
        _, qvals, _ = go_ai.montecarlo.pi_qval_from_actor_critic(states, forward_func)
        valid_moves = data.batch_valid_moves(states)
        min_qvals = np.min(qvals, axis=1, keepdims=True)
        qvals -= min_qvals
        qvals += 1e-7 * valid_moves
        qvals *= valid_moves

        target_pis = normalize(qvals ** 2, norm='l1')

        with tf.GradientTape() as tape:
            move_prob_distrs, state_vals = forward_pass(states, actor_critic, training=True)

            # Critic
            assert state_vals.shape == wins.shape
            critic_loss = mean_squared_error(wins, state_vals)
            tb_metrics['val_loss'].update_state(critic_loss)
            wins_01 = (np.copy(wins) + 1) / 2
            tb_metrics['pred_win_acc'].update_state(wins_01, state_vals > 0)

            # Actor
            actor_loss = cat_cross_entropy(target_pis, move_prob_distrs)
            tb_metrics['move_loss'].update_state(actor_loss)

            # Overall Loss
            overall_loss = critic_loss + 0 * actor_loss

        # compute and apply gradients
        gradients = tape.gradient(overall_loss, actor_critic.trainable_variables)
        optimizer.apply_gradients(zip(gradients, actor_critic.trainable_variables))

        # Metrics
        pbar.set_postfix_str('{:.1f}% ACC, {:.3f}VL, {:.3f}ML'.format(100 * tb_metrics['pred_win_acc'].result().numpy(),
                                                                      tb_metrics['val_loss'].result().numpy(),
                                                                      tb_metrics['move_loss'].result().numpy()))
    # Update the weights on disk
    actor_critic.save_weights(policy_args['model_path'])
