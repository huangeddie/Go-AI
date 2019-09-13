import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging
import random
from go_ai import go_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import mcts
import gym

gogame = gym.make('gym_go:go-v0', size=0).gogame


def make_actor_critic(board_size, critic_mode, critic_activation):
    action_size = board_size ** 2 + 1

    inputs = layers.Input(shape=(board_size, board_size, 6), name="board")
    valid_inputs = layers.Input(shape=(action_size,), name="valid_moves")
    invalid_values = layers.Input(shape=(action_size,), name="invalid_values")

    x = inputs

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)

    # Actor
    move_probs = layers.Conv2D(2, kernel_size=3, padding='same', activation='relu')(x)
    move_probs = layers.Flatten()(move_probs)
    move_probs = layers.Dense(action_size)(move_probs)
    move_probs = layers.Add()([move_probs, invalid_values])
    move_probs = layers.Softmax(name="move_probs")(move_probs)
    actor_out = move_probs

    # Critic
    move_vals = layers.Conv2D(2, kernel_size=3, padding='same', activation='relu')(x)
    move_vals = layers.Flatten()(move_vals)
    if critic_mode == 'q_net':
        move_vals = layers.Dense(action_size, activation=critic_activation)(move_vals)
        move_vals = layers.Multiply(name="move_vals")([move_vals, valid_inputs])
        critic_out = move_vals
    elif critic_mode == 'val_net':
        move_vals = layers.Dense(1, activation=critic_activation)(move_vals)
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
    invalid_moves = go_utils.get_invalid_moves(states)
    invalid_values = go_utils.get_invalid_values(states)
    valid_moves = 1 - invalid_moves
    return network([states.astype(np.float32),
                    valid_moves.astype(np.float32),
                    invalid_values.astype(np.float32)], training=training)


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

    for oriented_chunk in go_utils.all_orientations(chunk):
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
    replay_mem = list(zip(*replay_mem))
    states = np.array(list(replay_mem[0]), dtype=np.float32).transpose(0, 2, 3, 1)
    actions = np.array(list(replay_mem[1]), dtype=np.int)
    next_states = np.array(list(replay_mem[2]), dtype=np.float32).transpose(0, 2, 3, 1)
    rewards = np.array(list(replay_mem[3]), dtype=np.float32).reshape((-1,))
    terminals = np.array(list(replay_mem[4]), dtype=np.uint8)
    wins = np.array(list(replay_mem[5]), dtype=np.int)
    mct_pis = np.array(list(replay_mem[6]), dtype=np.float32)

    return states, actions, next_states, rewards, terminals, wins, mct_pis


def get_batch_obs(replay_mem, batch_size, index=None):
    '''
    Get a batch of orig_states, actions, states, rewards, terminals as np array out of replay memory
    '''

    # States were (BATCH_SIZE, 4, board_size, board_size)
    # Convert them to (BATCH_SIZE, board_size, board_size, 4)
    if index is None:
        batch = random.sample(replay_mem, batch_size)
    else:
        batch = replay_mem[index * batch_size: (index + 1) * batch_size]
    return replay_mem_to_numpy(batch)


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

    move_probs, move_vals = forward_pass(states, actor_critic, training=False)
    state_vals = tf.reduce_sum(move_probs * move_vals, axis=1)

    valid_moves = go_utils.get_valid_moves(states)

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
            go_utils.plot_move_distr('Critic', move_vals[i], valid_moves[i],
                                     scalar=state_vals[i].numpy())
        else:
            plt.subplot(num_states, num_cols, curr_col + num_cols * i)
            go_utils.plot_move_distr('MCTS', mcts_move_probs[i], valid_moves[i])
        curr_col += 1

        plt.subplot(num_states, num_cols, curr_col + num_cols * i)
        go_utils.plot_move_distr('Actor{}'.format(' Critic' if mcts_move_probs is not None else ''),
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


def sample_heatmaps(actor_critic, replay_mem, num_samples=2):
    states, actions, next_states, rewards, terminals, wins, mc_pis = get_batch_obs(replay_mem, batch_size=num_samples)
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
        states = state[np.newaxis].transpose(0, 2, 3, 1)
        invalid_moves = go_utils.get_invalid_moves(states)
        invalid_values = go_utils.get_invalid_values(states)
        valid_moves = 1 - invalid_moves
        move_probs, vals = policy([states, valid_moves, invalid_values])
        return move_probs[0], vals[0]

    return mcts_forward


def play_a_game(replay_mem, go_env, policy, max_steps, mc_sims):
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
        if num_steps > 30:
            temp = 0
        else:
            temp = 1
        pbar.set_postfix_str("{}T".format(temp))
        mcts_action_probs, _ = mct.get_action_probs(max_num_searches=mc_sims, temp=temp)
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

    black_won = 1 if info['area']['b'] > info['area']['w'] else -1

    # Add the last event to memory
    if replay_mem is not None:
        for turn, canonical_state, action, canonical_next_state, reward, done, mcts_action_probs in mem_cache:
            if turn == go_env.govars.BLACK:
                win = black_won
            else:
                win = -black_won
            add_to_replay_mem(replay_mem, canonical_state, action, canonical_next_state, reward, done, win, mcts_action_probs)

    # Game ended
    return num_steps


def reset_metrics(metrics):
    for key, metric in metrics.items():
        metric.reset_states()


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
        tf.summary.image("model heat maps", go_utils.plot_to_image(fig), step=step)


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


def evaluate(go_env, policy, opponent, max_steps, num_games, mc_sims):
    avg_metric = tf.keras.metrics.Mean()

    pbar = tqdm_notebook(range(num_games), desc='Evaluating against former self', leave=False)
    for episode in pbar:
        if episode % 2 == 0:
            black_won = pit(go_env, policy, opponent, max_steps, mc_sims)
            avg_metric(black_won)
        else:
            black_won = pit(go_env, opponent, policy, max_steps, mc_sims)
            avg_metric(-black_won)
        pbar.set_postfix_str('{:.1f}%'.format(100 * avg_metric.result()))

    return avg_metric.result()


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
