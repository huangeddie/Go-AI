import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym
import datetime
import logging
import random
import itertools
from go_ai import go_utils
import matplotlib.pyplot as plt
import collections
from functools import reduce
from tqdm import tqdm_notebook

def make_actor_critic(mode, board_size):
    inputs = layers.Input(shape=(board_size, board_size, 4), name="board")
    valid_inputs = layers.Input(shape=(board_size**2 + 1,), name="valid_moves")
    invalid_values = layers.Input(shape=(board_size**2 + 1,), name="invalid_values")
    
    x = layers.Flatten()(inputs)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    if mode == 'actor':
        move_probs = layers.Dense(128)(x)
        move_probs = layers.BatchNormalization()(move_probs)
        move_probs = layers.ReLU()(move_probs)
        move_probs = layers.Dense(50)(move_probs)
        move_probs = layers.Add()([move_probs, invalid_values])
        move_probs = layers.Softmax(name="move_probs")(move_probs)
        out = move_probs
    elif mode == 'q_net':
        move_vals = layers.Dense(128)(x)
        move_vals = layers.BatchNormalization()(move_vals)
        move_vals = layers.ReLU()(move_vals)
        move_vals = layers.Dense(50, activation='sigmoid')(move_vals)
        move_vals = layers.Multiply(name="move_vals")([move_vals, valid_inputs])
        out = move_vals
    elif mode == 'val_net':
        move_vals = layers.Dense(128)(x)
        move_vals = layers.BatchNormalization()(move_vals)
        move_vals = layers.ReLU()(move_vals)
        move_vals = layers.Dense(1, activation='sigmoid')(move_vals)
        out = move_vals
    else:
        raise Exception("Unknown mode")

    model = tf.keras.Model(inputs=[inputs, valid_inputs, invalid_values], outputs=out, name=mode)
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

def add_to_replay_mem(replay_mem, state, action_1d, next_state, reward, done):
    """
    Adds original event, plus augmented versions of those events
    """
    assert len(state.shape) == 3 and state.shape[1] == state.shape[2]
    board_size = state.shape[1]
    for s, a, ns in list(zip(go_utils.all_orientations(state, board_size), 
                             go_utils.all_orientations(action_1d, board_size), 
                             go_utils.all_orientations(next_state, board_size))):
        replay_mem.append((s, a, ns, reward, done))

def get_batch_obs(replay_mem, batch_size, index=None):
    '''
    Get a batch of orig_states, actions, states, rewards, terminals as np array out of replay memory
    '''
    
    # States were (BATCH_SIZE, 4, board_size, board_size)
    # Convert them to (BATCH_SIZE, board_size, board_size, 4)
    if index is None:
        batch = random.sample(replay_mem, batch_size)
    else:
        batch = replay_mem[index*batch_size: (index+1)*batch_size]
    batch = list(zip(*batch))
    states = np.array(list(batch[0]), dtype=np.float32).transpose(0,2,3,1)
    actions = np.array(list(batch[1]), dtype=np.int)
    next_states = np.array(list(batch[2]), dtype=np.float32).transpose(0,2,3,1)
    rewards = np.array(list(batch[3]), dtype=np.float32).reshape((-1,))
    terminals = np.array(list(batch[4]), dtype=np.uint8)
    
    return states, actions, next_states, rewards, terminals 

def state_responses(actor, critic, states, taken_actions, next_states, rewards, terminals):
    """
    Returns a figure of plots on the states and the models responses on those states
    """
    board_size = states[0].shape[0]
    
    move_probs = forward_pass(states, actor, training=False)
    move_vals = forward_pass(states, critic, training=False)
    state_vals = tf.reduce_sum(move_probs * move_vals, axis=1)
    
    valid_moves = go_utils.get_valid_moves(states)
    
    num_states = states.shape[0]
    num_cols = 4
    
    fig = plt.figure(figsize=(num_cols * 2.5, num_states * 2))
    for i in range(num_states):
        plt.subplot(num_states,num_cols,1 + num_cols*i)
        plt.axis('off')
        plt.title('Board')
        plt.imshow(states[i][:,:,[0,1,3]].astype(np.float))
        
        plt.subplot(num_states,num_cols, 2 + num_cols*i)
        if move_vals.shape[1] > 1:
            go_utils.plot_move_distr('Critic', 100*move_vals[i], valid_moves[i], 
                                     scalar=100*state_vals[i].numpy())
        else:
            plt.title('Critic')
            plt.bar(move_vals[i].numpy())

        plt.subplot(num_states,num_cols, 3 + num_cols*i)
        go_utils.plot_move_distr('Actor', 100*move_probs[i], valid_moves[i], 
                              scalar=None)
        
        plt.subplot(num_states,num_cols, 4 + num_cols*i)
        plt.axis('off')
        plt.title('Taken Action: {}\n{:.0f}R {}T'
                  .format(go_utils.action_1d_to_2d(taken_actions[i], board_size), 
                                                         rewards[i], terminals[i]))
        plt.imshow(next_states[i][:,:,[0,1,3]].astype(np.float))

    plt.tight_layout()
    return fig

def sample_heatmaps(actor, critic, replay_mem, num_samples=2):
    states, actions, next_states, rewards, terminals = get_batch_obs(replay_mem, batch_size=num_samples)
    assert len(states[0].shape) == 3 and states[0].shape[0] == states[0].shape[1], states[0].shape
    board_size = states[0].shape[0]
    
    # Add latest terminal state
    for (state, action, next_state, reward, terminal) in reversed(replay_mem):
        if terminal:
            states = np.append(states, state.transpose(1,2,0)[np.newaxis], axis=0)
            actions = np.append(actions, action)
            next_states = np.append(next_states, next_state.transpose(1,2,0)[np.newaxis], axis=0)
            rewards = np.append(rewards, reward)
            terminals = np.append(terminals, terminal)
            break
    # Add latest start state
    for (state, action, next_state, reward, terminal) in reversed(replay_mem):
        if np.sum(state[:2]) == 0:
            states = np.append(states, state.transpose(1,2,0)[np.newaxis], axis=0)
            actions = np.append(actions, action)
            next_states = np.append(next_states, next_state.transpose(1,2,0)[np.newaxis], axis=0)
            rewards = np.append(rewards, reward)
            terminals = np.append(terminals, terminal)
            break

    fig = state_responses(actor, critic, states, actions, next_states, rewards, terminals)
    return fig

def get_action(policy, state, epsilon=0):
    """
    Gets an action (1D) based on exploration/exploitation
    """
    
    if state.shape[0] == 4:
        # State shape will be (board_size, board_size, 4)
        # Note that we are assuming board_size to be greater than 4
        state = state.transpose(1,2,0)
    
    epsilon_choice = np.random.uniform()
    if epsilon_choice < epsilon:
        # Random move
        logging.debug("Exploring a random move")
        action = go_utils.random_action(state)
        
    else:
        # policy makes a move
        logging.debug("Exploiting policy's move")
        reshaped_state = state[np.newaxis].astype(np.float32)
        
        move_probs = forward_pass(reshaped_state, policy, training=False)
        action = go_utils.random_weighted_action(move_probs)
        
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

def play_a_game(replay_mem, go_env, black_policy, white_policy, max_steps, black_first=True):
    """
    Plays out a game, and adds the events to the given replay memory
    Returns the number of moves by the end of the game and the list 
    of rewards after every turn by the black player
    """
    
    board_size = go_env.board_size
    
    # Basic setup
    done = False
    num_steps = 0
    state = go_env.reset(black_first=black_first)
    
    # Make it a numpy array so it can be passed to the replay memory by reference
    # That way, all events of the same game will have the same reward
    win = np.zeros(1)
    
    if not black_first:
        # White moves first
        white_action = get_action(white_policy, state, 0 if white_policy is not None else 1)
        state, reward, done, info = go_env.step(go_utils.action_1d_to_2d(white_action, board_size))
        
        num_steps += 1
    
    while True:
        # Black move
        black_action = get_action(black_policy, state, epsilon=0 if black_policy is not None else 1)
        next_state, reward, done, info = go_env.step(go_utils.action_1d_to_2d(black_action, board_size))
        
        num_steps += 1      
            
        # Max number of steps or game ended by consecutive passing
        if done or num_steps >= max_steps:
            break
            
        # White move
        white_action = get_action(white_policy, next_state[[1,0,2,3]], epsilon=0 if white_policy is not None else 1)
        next_state, reward, done, info = go_env.step(go_utils.action_1d_to_2d(white_action, board_size))
        
        num_steps += 1
        
        # Max number of steps or game ended by consecutive passing
        if done or num_steps >= max_steps:
            break
            
        # Add to memory
        assert done == False
        if replay_mem is not None:
            add_to_replay_mem(replay_mem, state, black_action, next_state, win, done)
            
        # Setup for next event
        state = next_state
    
    # We're done
    done = True
    
    # Set the winner if we're done
    win[0] = int(info['area']['b'] > info['area']['w'])
    
    # Add the last event to memory
    if replay_mem is not None:
        add_to_replay_mem(replay_mem, state, black_action, next_state, win, done)
    
    # Game ended
    return win.item(), num_steps

def reset_metrics(metrics):
    for key, metric in metrics.items():
        metric.reset_states()

def log_to_tensorboard(summary_writer, metrics, step, replay_mem, actor, critic):
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
        fig = sample_heatmaps(actor, critic, replay_mem, num_samples=2)
        tf.summary.image("model heat maps", go_utils.plot_to_image(fig), step=step)
        
def evaluate(go_env, actor, opponent, level_paths, max_steps):
    avg_metric = tf.keras.metrics.Mean()
    for level_idx, level_path in tqdm_notebook(enumerate(level_paths), desc='Evaluating'):
        if level_path is not None:
            opponent.load_weights(level_path)

        white_policy = opponent if level_path is not None else None

        for black_first in [True, False]:
            avg_metric.reset_states()
            for episode in tqdm_notebook(range(128), desc='Evaluating against level {} opponent'.format(level_idx)):
                black_won, _ = play_a_game(None, go_env, black_policy=actor, white_policy=white_policy, max_steps=max_steps,
                                           black_first=black_first)
                avg_metric(black_won)

            print('{}L_{}: {:.1f}%'.format(level_idx, 'B' if black_first else 'W', 100*avg_metric.result().numpy()))
            
def play_against(policy, go_env):
    state = go_env.reset()

    done = False
    while not done:
        go_env.render()

        # Actor's move
        action = get_action(policy, state, epsilon=0)

        state, reward, done, info = go_env.step(go_utils.action_1d_to_2d(action, go_env.board_size))
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