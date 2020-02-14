import collections
import os
import pickle
import random

import gym
import numpy as np
from mpi4py import MPI

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


def batch_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    """
    assert len(states.shape) == 4
    batchsize = states.shape[0]
    board_size = states.shape[2]
    invalid_moves = states[:, GoVars.INVD_CHNL].reshape((batchsize, -1))
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


def replay_to_events(replay):
    trans_trajs = []
    for traj in replay:
        trans_trajs.extend(traj.get_events())
    return trans_trajs


def events_to_numpy(events):
    if len(events) == 0:
        return [], [], [], [], [], []
    unzipped = list(zip(*events))

    states = np.array(list(unzipped[0]), dtype=np.float32)
    actions = np.array(list(unzipped[1]), dtype=np.int)
    rewards = np.array(list(unzipped[2]), dtype=np.float32).reshape((-1,))
    next_states = np.array(list(unzipped[3]), dtype=np.float32)
    terminals = np.array(list(unzipped[4]), dtype=np.uint8)
    wins = np.array(list(unzipped[5]), dtype=np.int)
    pis = np.array(list(unzipped[6]), dtype=np.float32)

    return states, actions, rewards, next_states, terminals, wins, pis


def load_replay(replay_path):
    """
    Loads replay data from a directory.
    :param replay_path:
    :param worker_rank: If specified, loads only that specific worker's data. Otherwise it loads all data from all workers
    :return:
    """
    with open(replay_path, 'rb') as f:
        replay = pickle.load(f)
    return replay


def mpi_sample_eventdata(comm: MPI.Intracomm, replay_path, batches, batchsize):
    """
    :param replay_path:
    :param batches:
    :param batchsize:
    :return: Batches of sample data, len of total data that was sampled
    """
    # Workers sample data one at a time to avoid memory issues
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    sample_data = None
    replay_len = None
    for worker in range(world_size):
        if rank == worker:
            replay = load_replay(replay_path)
            replay_len = len(replay)
            # Seperate into black wins and black non-wins to ensure even sampling between the two
            black_wins = list(filter(lambda traj: traj.get_winner() == 1, replay))
            black_nonwins = list(filter(lambda traj: traj.get_winner() != 1, replay))
            black_wins = replay_to_events(black_wins)
            black_nonwins = replay_to_events(black_nonwins)
            n = min(len(black_wins), len(black_nonwins))
            sample_size = min(batchsize * batches // 2, n)
            sample_data = random.sample(black_wins, sample_size) + random.sample(black_nonwins, sample_size)
            # Save memory
            del replay
        comm.Barrier()

    random.shuffle(sample_data)
    sample_data = events_to_numpy(sample_data)

    sample_size = len(sample_data[0])
    for component in sample_data:
        assert len(component) == sample_size

    splits = max(sample_size // batchsize, 1)
    batched_sampledata = [np.array_split(component, splits) for component in sample_data]
    batched_sampledata = list(zip(*batched_sampledata))

    return batched_sampledata, replay_len


def mpi_disk_append_replay(comm: MPI.Intracomm, args, replays):
    rank = comm.Get_rank()
    for worker in range(comm.Get_size()):
        if rank == worker:
            if os.path.exists(args.replay_path):
                all_replays = load_replay(args.replay_path)
                all_replays.extend(replays)
            else:
                all_replays = replays

            with open(args.replay_path, 'wb') as f:
                pickle.dump(all_replays, f)
        comm.Barrier()


def reset_replay(args):
    if os.path.exists(args.replay_path):
        os.remove(args.replay_path)
    replay_buffer = collections.deque(maxlen=args.replaysize)
    with open(args.replay_path, 'wb') as f:
        pickle.dump(replay_buffer, f)
