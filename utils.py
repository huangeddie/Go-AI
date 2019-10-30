import argparse
import os
import sys

import torch
from mpi4py import MPI
from tqdm import tqdm

from go_ai.models import value_models


def hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=bool, default=False, help='continue from checkpoint')

    parser.add_argument('--boardsize', type=int, help='board size')
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches')

    parser.add_argument('--starttemp', type=float, default=1 / 16, help='initial temperature')
    parser.add_argument('--tempdecay', type=float, default=3 / 4, help='temperature decay')
    parser.add_argument('--mintemp', type=float, default=1 / 16, help='minimum temperature')

    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--replaysize', type=int, default=500000, help='replay memory size')
    parser.add_argument('--trainstep-size', type=int, default=1000 * 32, help='train step size')

    parser.add_argument('--iterations', type=int, default=128, help='iterations')
    parser.add_argument('--episodes', type=int, default=128, help='episodes')
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
    comm.allgather(None)
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
        tqdm.write(s, file=sys.stderr)


def sync_data(rank, comm: MPI.Intracomm, args):
    if rank == 0:
        if args.checkpoint:
            assert os.path.exists(args.check_path)
        else:
            # Clear worker data
            episode_files = os.listdir(args.episodes_dir)
            for item in episode_files:
                if item.endswith(".pickle"):
                    os.remove(os.path.join(args.episodes_dir, item))
            # Set parameters
            new_model = value_models.ValueNet(args.boardsize)
            torch.save(new_model.state_dict(), args.check_path)
    parallel_err(rank, "Continuing from checkpoint: {}".format(args.checkpoint))
    comm.allgather(None)
