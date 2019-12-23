import argparse
import datetime
import os
import shutil

import torch
from mpi4py import MPI

from go_ai import data, policies, parallel
from go_ai.models import value, actorcritic


def hyperparameters():
    today = str(datetime.date.today())

    parser = argparse.ArgumentParser()

    # Go Environment
    parser.add_argument('--boardsize', type=int, default=9, help='board size')
    parser.add_argument('--reward', type=str, choices=['real', 'heuristic'], default='real', help='reward system')

    # Monte Carlo Tree Search
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    # Exploration
    parser.add_argument('--temp', type=float, default=1, help='initial temperature')
    parser.add_argument('--tempsteps', type=int, default=24, help='first k steps to apply temperature to pi')

    # Data Sizes
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--replaysize', type=int, default=200000, help='max replay memory size')
    parser.add_argument('--trainsize', type=int, default=1000 * 32, help='train data size for one iteration')

    # Training
    parser.add_argument('--baseline', type=bool, default=False, help='load baseline model')
    parser.add_argument('--iterations', type=int, default=128, help='iterations')
    parser.add_argument('--episodes', type=int, default=32, help='episodes')
    parser.add_argument('--evaluations', type=int, default=32, help='episodes')
    parser.add_argument('--eval-interval', type=int, default=1, help='iterations per evaluation')

    # Disk Data
    parser.add_argument('--episodesdir', type=str, default='bin/episodes/', help='directory to store episodes')
    parser.add_argument('--savedir', type=str, default=f'bin/checkpoints/{today}/')

    # Model
    parser.add_argument('--model', type=str, choices=['val', 'ac', 'rand', 'greedy', 'human'], default='val',
                        help='type of model')
    parser.add_argument('--resblocks', type=int, default=4, help='number of basic blocks for resnets')

    # Hardware
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device for pytorch models')

    # Other
    parser.add_argument('--render', type=str, choices=['terminal', 'human'], default='terminal',
                        help='type of rendering')

    return parser.parse_args()


def sync_checkpoint(comm: MPI.Intracomm, args, new_pi, old_pi):
    rank = comm.Get_rank()
    checkpath = get_modelpath(args, 'checkpoint')
    if rank == 0:
        torch.save(new_pi.pytorch_model.state_dict(), checkpath)
    comm.Barrier()
    # Update other policy
    old_pi.pytorch_model.load_state_dict(torch.load(checkpath))


def sync_data(comm: MPI.Intracomm, args):
    rank = comm.Get_rank()
    if rank == 0:
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)

        checkpath = get_modelpath(args, 'checkpoint')
        if args.baseline:
            baseline_path = get_modelpath(args, 'baseline')
            shutil.copy(baseline_path, checkpath)
            parallel.parallel_debug(comm, "Starting from baseline")
        else:
            # Clear worker data
            episodesdir = args.episodesdir
            data.clear_episodesdir(episodesdir)
            # Save new model
            new_model, _ = create_model(args, '', latest_checkpoint=False)

            torch.save(new_model.state_dict(), checkpath)
            parallel.parallel_debug(comm, "Starting from scratch")

    comm.Barrier()


def get_modelpath(args, savetype):
    if savetype == 'checkpoint':
        dir = args.savedir
    elif savetype == 'baseline':
        dir = 'bin/baselines/'
    else:
        raise Exception(f"Unknown location type: {savetype}")
    path = os.path.join(dir, f'{args.model}{args.boardsize}.pt')

    return path


def create_model(args, name, baseline=False, latest_checkpoint=False, checkdir=None):
    model = args.model
    size = args.boardsize
    if model == 'val':
        net = value.ValueNet(size, args.resblocks)
        pi = policies.Value(name, net, args.mcts, args.temp, args.tempsteps)
    elif model == 'ac':
        net = actorcritic.ActorCriticNet(size, args.resblocks)
        pi = policies.ActorCritic(name, net, args.mcts, args.temp, args.tempsteps)
    elif model == 'rand':
        net = None
        pi = policies.RAND_PI
    elif model == 'greedy':
        net = None
        pi = policies.GREEDY_PI
    elif model == 'human':
        net = None
        pi = policies.Human(args.render)
    else:
        raise Exception("Unknown model argument", model)

    if baseline:
        assert not latest_checkpoint
        net.load_state_dict(torch.load(f'bin/baselines/{model}{size}.pt'))
    elif latest_checkpoint:
        assert not baseline
        assert checkdir is None
        latest_checkpath = get_modelpath(args, 'checkpoint')
        net.load_state_dict(torch.load(latest_checkpath))
    elif checkdir is not None:
        assert not latest_checkpoint
        checkpath = os.path.join(checkdir, f'{model}{size}.pt')
        net.load_state_dict(torch.load(checkpath))

    return net, pi
