import collections
import os
from datetime import datetime

import gym
import torch
from mpi4py import MPI

from go_ai import data, utils
from go_ai.policies import baselines


def model_eval(comm, args, curr_pi, checkpoint_pi, winrates):
    go_env = gym.make('gym_go:go-v0', size=args.size, reward_method=args.reward)
    # See how this new model compares
    for opponent in [checkpoint_pi, baselines.RAND_PI, baselines.GREEDY_PI]:
        # Play some games
        utils.mpi_log_debug(comm, f'Pitting {curr_pi} V {opponent}')
        wr, _, _ = utils.mpi_play(comm, go_env, curr_pi, opponent, args.evaluations)
        winrates[opponent] = wr


def train_step(comm, args, curr_pi, optim, checkpoint_pi):
    # Environment
    go_env = gym.make('gym_go:go-v0', size=args.size, reward_method=args.reward)
    curr_model = curr_pi.pytorch_model

    # Play episodes
    utils.mpi_log_debug(comm, f'Self-Playing {checkpoint_pi} V {checkpoint_pi}...')
    _, _, replays = utils.mpi_play(comm, go_env, checkpoint_pi, checkpoint_pi, args.episodes)

    # Write episodes
    data.mpi_disk_append_replay(comm, args, replays)
    utils.mpi_log_debug(comm, 'Added all replay data to disk')

    # Sample data as batches
    traindata, replay_len = data.mpi_sample_eventdata(comm, args.replay_path, args.batches, args.batchsize)

    # Optimize
    utils.mpi_log_debug(comm, f'Optimizing in {len(traindata)} training steps...')
    metrics = curr_model.optimize(comm, traindata, optim)

    # Sync model
    utils.mpi_log_debug(comm, f'Optimized | {str(metrics)}')

    return metrics, replay_len


def train(comm, args, curr_pi, checkpoint_pi):
    # Optimizer
    curr_model = curr_pi.pytorch_model
    optim = torch.optim.Adam(curr_model.parameters(), args.lr, weight_decay=1e-4)

    # Timer
    starttime = datetime.now()

    # Header output
    utils.mpi_log_info(comm, "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tC_WR\tR_WR\tG_WR")

    winrates = collections.defaultdict(float)
    for iteration in range(args.iterations):
        # Train
        metrics, replay_len = train_step(comm, args, curr_pi, optim, checkpoint_pi)

        # Model Evaluation
        model_eval(comm, args, curr_pi, checkpoint_pi, winrates)

        # Sync
        utils.mpi_sync_checkpoint(comm, args, new_pi=curr_pi, old_pi=checkpoint_pi)

        # Print iteration summary
        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = f"{str(delta).split('.')[0]}\t{iteration:02d}\t{replay_len:07d}\t" \
                    f"{100 * metrics.crit_acc:04.1f}\t{metrics.crit_loss:04.3f}\t" \
                    f"{100 * metrics.act_acc:04.1f}\t{metrics.act_loss:04.3f}\t" \
                    f"{100 * winrates[checkpoint_pi]:04.1f}\t{100 * winrates[baselines.RAND_PI]:04.1f}\t" \
                    f"{100 * winrates[baselines.GREEDY_PI]:04.1f}"
        utils.mpi_log_info(comm, iter_info)


if __name__ == '__main__':
    # Parallel Setup
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    # Arguments
    args = utils.hyperparameters()

    # Save directory
    if not os.path.exists(args.checkdir):
        if comm.Get_rank() == 0:
            os.mkdir(args.checkdir)
    comm.Barrier()

    # Logging
    utils.mpi_config_log(args, comm)
    utils.mpi_log_debug(comm, f"{world_size} Workers, {args}")

    # Set parameters and replay data on disk
    utils.mpi_sync_data(comm, args)

    # Model and Policies
    curr_pi, curr_model = baselines.create_policy(args, 'Current')
    checkpoint_pi, checkpoint_model = baselines.create_policy(args, 'Checkpoint')

    # Device
    device = torch.device(args.device)
    curr_model.to(device)
    checkpoint_model.to(device)

    # Train
    train(comm, args, curr_pi, checkpoint_pi)
