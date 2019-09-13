import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging
import random
import mcts
import gym

go_env = gym.make('gym_go:go-v0', size=0)
govars = go_env.govars
gogame = go_env.gogame


def get_valid_moves(states):
    return 1 - get_invalid_moves(states)


def get_invalid_values(states):
    """
    Returns the action values of the states where invalid moves have -infinity value (minimum value of float32)
    and valid moves have 0 value
    """
    invalid_moves = get_invalid_moves(states)
    invalid_values = np.finfo(np.float32).min * invalid_moves
    return invalid_values


def get_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    Assumes shape to be [BATCH SIZE, BOARD SIZE, BOARD SIZE, 6]
    """
    assert len(states.shape) == 4
    batch_size = states.shape[0]
    board_size = states.shape[1]
    invalid_moves = states[:, :, :, govars.INVD_CHNL].reshape((batch_size, -1))
    invalid_moves = np.insert(invalid_moves, board_size ** 2, 0, axis=1)
    return invalid_moves


def make_actor_critic(board_size, critic_mode, critic_activation):
    action_size = board_size ** 2 + 1

    inputs = layers.Input(shape=(board_size, board_size, 6), name="board")
    valid_inputs = layers.Input(shape=(action_size,), name="valid_moves")
    invalid_values = layers.Input(shape=(action_size,), name="invalid_values")

    x = inputs

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    # Actor
    move_probs = layers.Conv2D(2, kernel_size=3, padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    move_probs = layers.Flatten()(move_probs)
    move_probs = layers.Dense(action_size, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(move_probs)
    move_probs = layers.Add()([move_probs, invalid_values])
    move_probs = layers.Softmax(name="move_probs")(move_probs)
    actor_out = move_probs

    # Critic
    move_vals = layers.Conv2D(2, kernel_size=3, padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    move_vals = layers.Flatten()(move_vals)
    if critic_mode == 'q_net':
        move_vals = layers.Dense(action_size, activation=critic_activation,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(move_vals)
        move_vals = layers.Multiply(name="move_vals")([move_vals, valid_inputs])
        critic_out = move_vals
    elif critic_mode == 'val_net':
        move_vals = layers.Dense(1, activation=critic_activation,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(move_vals)
        critic_out = move_vals
    else:
        raise Exception("Unknown critic mode")

    model = tf.keras.Model(inputs=[inputs, valid_inputs, invalid_values], outputs=[actor_out, critic_out],
                           name='actor_critic')
    return model


def forward_pass(states, network, training):
    """
    Since the neural nets take in more than one parameter,
    this functions serves as a wrapper to forward pass the data through the networks
    """
    invalid_moves = get_invalid_moves(states)
    invalid_values = get_invalid_values(states)
    valid_moves = 1 - invalid_moves
    return network([states.astype(np.float32),
                    valid_moves.astype(np.float32),
                    invalid_values.astype(np.float32)], training=training)


def get_action(policy, state, epsilon=0):
    """
    Gets an action (1D) based on exploration/exploitation
    """

    if state.shape[0] == 6:
        # State shape will be (board_size, board_size, 6)
        # Note that we are assuming board_size to be greater than 6
        state = state.transpose(1, 2, 0)

    epsilon_choice = np.random.uniform()
    if epsilon_choice < epsilon:
        # Random move
        logging.debug("Exploring a random move")
        action = gogame.random_action(state.reshape(2, 0, 1))

    else:
        # policy makes a move
        logging.debug("Exploiting policy's move")
        reshaped_state = state[np.newaxis].astype(np.float32)

        move_probs, _ = forward_pass(reshaped_state, policy, training=False)
        action = gogame.random_weighted_action(move_probs[0])

    return action


def get_values_for_actions(move_val_distrs, actions):
    '''
    Actions should be a one hot array [batch size, ] array
    Get value from board_values based on action, or take the passing_values if the action is None
    '''
    one_hot_actions = tf.one_hot(actions, depth=move_val_distrs.shape[1])
    assert move_val_distrs.shape == one_hot_actions.shape
    one_hot_move_values = move_val_distrs * one_hot_actions
    move_values = tf.reduce_sum(one_hot_move_values, axis=1)
    return move_values


def make_mcts_forward(policy):
    def mcts_forward(state):
        states = state.transpose(0, 2, 3, 1)
        move_probs, vals = forward_pass(states, policy, training=False)
        return move_probs, vals

    return mcts_forward


def add_to_replay_mem(replay_mem, state, action_1d, next_state, reward, done, win, mcts_action_probs):
    """
    Adds original event, plus augmented versions of those events
    States are assumed to be (6, BOARD_SIZE, BOARD_SIZE)
    """
    assert len(state.shape) == 3 and state.shape[1] == state.shape[2]
    board_size = state.shape[1]
    num_channels = state.shape[0]

    if action_1d < board_size ** 2:
        action_2d_one_hot = np.eye(board_size ** 2)[action_1d].reshape(1, board_size, board_size)
    else:
        action_2d_one_hot = np.zeros((1, board_size, board_size))

    mc_board_pi = mcts_action_probs[:-1].reshape(1, board_size, board_size)

    chunk = np.concatenate([state, action_2d_one_hot, next_state, mc_board_pi], axis=0)

    for oriented_chunk in gogame.get_symmetries(chunk):
        s = oriented_chunk[:num_channels]
        a = oriented_chunk[num_channels]
        if np.count_nonzero(a) == 0:
            a = board_size ** 2
        else:
            a = np.argmax(a)
        ns = oriented_chunk[num_channels + 1:(num_channels + 1) + num_channels]
        assert (num_channels + 1) + num_channels == oriented_chunk.shape[0] - 1
        mc_pi = oriented_chunk[-1].flatten()
        mc_pi = np.append(mc_pi, mcts_action_probs[-1])
        replay_mem.append((s, a, ns, reward, done, win, mc_pi))


def replay_mem_to_numpy(replay_mem):
    """
    Turns states from (BATCH_SIZE, 6, BOARD_SIZE, BOARD_SIZE) to (BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 6)
    :param replay_mem:
    :return:
    """
    replay_mem = list(zip(*replay_mem))
    states = np.array(list(replay_mem[0]), dtype=np.float32).transpose(0, 2, 3, 1)
    actions = np.array(list(replay_mem[1]), dtype=np.int)
    next_states = np.array(list(replay_mem[2]), dtype=np.float32).transpose(0, 2, 3, 1)
    rewards = np.array(list(replay_mem[3]), dtype=np.float32).reshape((-1,))
    terminals = np.array(list(replay_mem[4]), dtype=np.uint8)
    wins = np.array(list(replay_mem[5]), dtype=np.int)
    mct_pis = np.array(list(replay_mem[6]), dtype=np.float32)

    return states, actions, next_states, rewards, terminals, wins, mct_pis

def self_play(replay_mem, go_env, policy, max_steps, mc_sims):
    """
    Plays out a game, by pitting the policy against itself,
    and adds the events to the given replay memory

    Returns the number of moves by the end of the game and the list
    of rewards after every turn by the black player
    """

    # Basic setup
    num_steps = 0
    state = go_env.reset()

    mem_cache = []

    mcts_forward = make_mcts_forward(policy)
    mct = mcts.MCTree(state, mcts_forward)

    while True:
        # Get turn
        curr_turn = go_env.turn

        # Get canonical state for policy and memory
        canonical_state = go_env.gogame.get_canonical_form(state, curr_turn)

        # Get action from MCT
        mcts_action_probs, _ = mct.get_action_probs(max_num_searches=mc_sims, temp=1/16)
        action = gogame.random_weighted_action(mcts_action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, info = go_env.step(action)
        mct.step(action)

        # Get canonical form of next state for memory
        canonical_next_state = go_env.gogame.get_canonical_form(next_state, curr_turn)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        mem_cache.append((curr_turn, canonical_state, action, canonical_next_state, reward, done, mcts_action_probs))

        # Increment steps
        num_steps += 1

        # Max number of steps or game ended by consecutive passing
        if done:
            break

        # Setup for next event
        state = next_state

    assert done

    if info['area']['b'] > info['area']['w']:
        black_won = 1
    elif info['area']['b'] < info['area']['w']:
        black_won = -1
    else:
        black_won = 0

    # Add the last event to memory
    if replay_mem is not None:
        for turn, canonical_state, action, canonical_next_state, reward, done, mcts_action_probs in mem_cache:
            if turn == go_env.govars.BLACK:
                win = black_won
            else:
                win = -black_won
            add_to_replay_mem(replay_mem, canonical_state, action, canonical_next_state, reward, done, win,
                              mcts_action_probs)

    # Game ended
    return num_steps


def pit(go_env, black_policy, white_policy, max_steps, mc_sims):
    num_steps = 0
    state = go_env.reset()

    black_forward = make_mcts_forward(black_policy)
    white_forward = make_mcts_forward(white_policy)
    black_mct = mcts.MCTree(state, black_forward)
    white_mct = mcts.MCTree(state, white_forward)

    while True:
        # Get turn
        curr_turn = go_env.turn

        # Get an action
        if curr_turn == go_env.govars.BLACK:
            mcts_action_probs, _ = black_mct.get_action_probs(max_num_searches=mc_sims, temp=0)
        else:
            assert curr_turn == go_env.govars.WHITE
            mcts_action_probs, _ = white_mct.get_action_probs(max_num_searches=mc_sims, temp=0)
        action = gogame.random_weighted_action(mcts_action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, info = go_env.step(action)

        # Sync the MC Trees
        black_mct.step(action)
        white_mct.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Increment steps
        num_steps += 1

        # Max number of steps or game ended by consecutive passing
        if done:
            break

    assert done

    if info['area']['b'] > info['area']['w']:
        black_won = 1
    elif info['area']['b'] < info['area']['w']:
        black_won = -1
    else:
        assert info['area']['b'] == info['area']['w']
        black_won = 0

    return black_won


def play_against(policy, go_env):
    state = go_env.reset()

    done = False
    while not done:
        go_env.render()

        # Actor's move
        action = get_action(policy, state, epsilon=0)

        state, reward, done, info = go_env.step(action)
        go_env.render()

        # Player's move
        player_moved = False
        while not player_moved:
            coords = input("Enter coordinates separated by space (`q` to quit)\n")
            if coords == 'q':
                done = True
                break
            if coords == 'r':
                go_env.reset()
                break
            if coords == 'p':
                go_env.step(None)
                break
            coords = coords.split()
            try:
                row = int(coords[0])
                col = int(coords[1])
                print(row, col)
                state, reward, done, info = go_env.step((row, col))
                player_moved = True
            except Exception as e:
                print(e)


