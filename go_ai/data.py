import os

import gym
import numpy as np

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


def batch_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    """
    assert len(states.shape) == 4
    batch_size = states.shape[0]
    board_size = states.shape[2]
    invalid_moves = states[:, GoVars.INVD_CHNL].reshape((batch_size, -1))
    invalid_moves = np.insert(invalid_moves, board_size ** 2, 0, axis=1)
    return invalid_moves


def batch_valid_moves(states):
    return 1 - batch_invalid_moves(states)


def batch_invalid_values(states):
    """
    Returns the action values of the states where invalid moves have -infinity value (minimum value of float32)
    and valid moves have 0 value
    """
    invalid_moves = batch_invalid_moves(states)
    invalid_values = np.finfo(np.float32).min * invalid_moves
    return invalid_values


def batch_random_symmetries(states):
    assert len(states.shape) == 4
    processed_states = []
    for state in states:
        processed_states.append(GoGame.random_symmetry(state))
    return np.array(processed_states)


def replaylist_to_numpy(replay_mem):
    """
    Turns states from (BATCH_SIZE, 6, BOARD_SIZE, BOARD_SIZE) to (BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 6)
    :param replay_mem:
    :return:
    """
    replay_mem = list(zip(*replay_mem))

    states = np.array(list(replay_mem[0]), dtype=np.float32)
    actions = np.array(list(replay_mem[1]), dtype=np.int)
    next_states = np.array(list(replay_mem[2]), dtype=np.float32)
    rewards = np.array(list(replay_mem[3]), dtype=np.float32).reshape((-1,))
    terminals = np.array(list(replay_mem[4]), dtype=np.uint8)
    wins = np.array(list(replay_mem[5]), dtype=np.int)

    return states, actions, next_states, rewards, terminals, wins


def episodes_from_dir(episodes_dir):
    """
    Loades episode data from a given directory
    episode_dir ->
        * worker_0.npz
        * worker_1.npz
        * ...

    worker_n.npz are files in the format [states, actions, next_states, rewards, terminals, wins]
    :param episodes_dir:
    :return: all of [states, actions, next_states, rewards, terminals, wins] concatenated shuffled
    """
    worker_data = []
    for file in os.listdir(episodes_dir):
        if file[0] == '.':
            continue
        data_batch = np.load(os.path.join(episodes_dir, file))
        worker_data.append([data_batch[file] for file in data_batch.files])

    # Worker data is a list.
    # Each element of the list is in the form
    # [states, actions, next_states, rewards, terminals, wins]
    batched_data = list(zip(*worker_data))
    # Batched data is in the form
    # [[[batch of states],...,[batch of states]], [[batch of actions],...],...]

    combined_data = []
    for batches in batched_data:
        combined_data.append(np.concatenate(batches))

    # Shuffle
    data_len = len(combined_data[0])
    perm = np.random.permutation(data_len)
    for i in range(len(combined_data)):
        assert len(combined_data[i]) == data_len
        combined_data[i] = combined_data[i][perm]

    return combined_data
