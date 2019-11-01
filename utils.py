import argparse
import os
import sys

import torch
from mpi4py import MPI
from tqdm import tqdm

from go_ai import data
from go_ai.models import value_models
import time


def hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=bool, default=False, help='continue from checkpoint')

    parser.add_argument('--boardsize', type=int, help='board size')
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches')

    parser.add_argument('--temp', type=float, default=1 / 8, help='initial temperature')
    parser.add_argument('--tempsteps', type=float, default=8, help='first k steps to apply temperature to pi')

    parser.add_argument('--batchsize', type=int, default=64, help='batch size')
    parser.add_argument('--replaysize', type=int, default=400000, help='replay memory size')
    parser.add_argument('--trainstep-size', type=int, default=1000 * 64, help='train step size')

    parser.add_argument('--iterations', type=int, default=128, help='iterations')
    parser.add_argument('--episodes', type=int, default=256, help='episodes')
    parser.add_argument('--evaluations', type=int, default=128, help='episodes')
    parser.add_argument('--eval-interval', type=int, default=1, help='iterations per evaluation')

    parser.add_argument('--episodes-dir', type=str, default='episodes/', help='directory to store episodes')
    parser.add_argument('--check-path', type=str, default='checkpoints/checkpoint.pt', help='model path for checkpoint')
    parser.add_argument('--tmp-path', type=str, default='checkpoints/tmp.pt', help='model path for temp model')
    parser.add_argument('--demotraj-path', type=str, help='path for sample trajectory')

    return parser.parse_args()


def sync_checkpoint(rank, comm: MPI.Intracomm, newcheckpoint_pi, check_path, other_pi):
    if rank == 0:
        torch.save(newcheckpoint_pi.pytorch_model.state_dict(), check_path)
    comm.Barrier()
    # Update other policy
    other_pi.pytorch_model.load_state_dict(torch.load(check_path))


def parallel_out(rank, s):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    if rank == 0:
        print(s, flush=True)


def parallel_err(rank, s):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    if rank == 0:
        tqdm.write(f"{time.strftime('%H:%M:%S', time.localtime())}\t{s}", file=sys.stderr)


def sync_data(rank, comm: MPI.Intracomm, args):
    if rank == 0:
        if args.checkpoint:
            assert os.path.exists(args.check_path)
        else:
            # Clear worker data
            episodes_dir = args.episodes_dir
            data.clear_episodes_dir(episodes_dir)
            # Set parameters
            new_model = value_models.ValueNet(args.boardsize)
            torch.save(new_model.state_dict(), args.check_path)
    parallel_err(rank, "Using checkpoint: {}".format(args.checkpoint))
    comm.Barrier()
