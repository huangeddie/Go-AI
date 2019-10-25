import argparse
import os
import sys

import torch
from tqdm import tqdm

from go_ai.models import value_models


def hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spawnmethod', type=str, help='spawn method for multiprocessing')
    parser.add_argument('--workers', type=int, help='number of parallel workers')
    parser.add_argument('--checkpoint', type=bool, default=False, help='continue from checkpoint')

    parser.add_argument('--boardsize', type=int, help='board size')
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches')

    parser.add_argument('--starttemp', type=float, default=1 / 32, help='initial temperature')
    parser.add_argument('--tempdecay', type=float, default=3 / 4, help='temperature decay')
    parser.add_argument('--mintemp', type=float, default=1 / 128, help='minimum temperature')

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
    parser.add_argument('--demotraj-path', type=str, default='episodes/a_traj.pdf', help='path for sample trajectory')

    return parser.parse_args()


def sync_checkpoint(rank, barrier, curr_pi, check_path, checkpoint_pi):
    if rank == 0:
        torch.save(curr_pi.pytorch_model.state_dict(), check_path)
    barrier.wait()
    # Update checkpoint policy
    checkpoint_pi.pytorch_model.load_state_dict(torch.load(check_path))


def parallel_print(rank, s):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    if rank == 0:
        print(s)


def setup(args, barrier, rank):
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
        tqdm.write("Continuing from checkpoint: {}".format(args.checkpoint), file=sys.stderr)
    barrier.wait()
