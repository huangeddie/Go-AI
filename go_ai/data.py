import os
import pickle

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


def load_replaydata(episodes_dir, worker_rank=None):
    all_data = []
    files = os.listdir(episodes_dir)
    for file in files:
        if '.pickle' in file:
            if worker_rank is not None and str(worker_rank) not in file:
                continue
            with open(episodes_dir + file, 'rb') as f:
                worker_data = pickle.load(f)
                all_data.extend(worker_data)
    return all_data

def save_replaydata(replay_data, episodes_dir, worker_rank):
    outpath = os.path.join(episodes_dir, "worker_{}.pickle".format(worker_rank))
    with open(outpath, 'wb') as f:
        pickle.dump(replay_data, f)


def clear_episodes_dir(episodes_dir):
    episode_files = os.listdir(episodes_dir)
    for item in episode_files:
        if item.endswith(".pickle"):
            os.remove(os.path.join(episodes_dir, item))