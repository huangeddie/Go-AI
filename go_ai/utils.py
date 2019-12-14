import argparse
import datetime
import os

import torch
from mpi4py import MPI

from go_ai import data, policies
from go_ai.models import value, actorcritic
from go_ai.parallel import parallel_err


def hyperparameters():
    today = str(datetime.date.today())

    parser = argparse.ArgumentParser()

    # Go Environment
    parser.add_argument('--boardsize', type=int, default=9, help='board size')
    parser.add_argument('--reward', type=str, choices=['real', 'heuristic'], default='real', help='reward system')

    # Monte Carlo Tree Search
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches')
    parser.add_argument('--branches', type=int, default=4, help='branch degree for searching')
    parser.add_argument('--depth', type=int, default=3, help='search depth')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    # Exploration
    parser.add_argument('--temp', type=float, default=1 / 10, help='initial temperature')
    parser.add_argument('--tempsteps', type=float, default=8, help='first k steps to apply temperature to pi')

    # Data Sizes
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--replaysize', type=int, default=200000, help='max replay memory size')
    parser.add_argument('--trainsize', type=int, default=1000 * 32, help='train data size for one iteration')

    # Training
    parser.add_argument('--checkpoint', type=bool, default=False, help='continue from checkpoint')
    parser.add_argument('--iterations', type=int, default=128, help='iterations')
    parser.add_argument('--episodes', type=int, default=32, help='episodes')
    parser.add_argument('--evaluations', type=int, default=32, help='episodes')
    parser.add_argument('--eval-interval', type=int, default=1, help='iterations per evaluation')

    # Disk Data
    parser.add_argument('--episodesdir', type=str, default='bin/episodes/', help='directory to store episodes')
    parser.add_argument('--savedir', type=str, default=f'bin/baselines/{today}/')
    parser.add_argument('--basepath', type=str, default=f'bin/{today}/base.pt', help='model path for baseline model')

    # Model
    parser.add_argument('--agent', type=str, choices=['mcts', 'ac', 'mcts-ac'], default='mcts',
                        help='type of agent/model')
    parser.add_argument('--baseagent', type=str, choices=['mcts', 'ac', 'mcts-ac', 'rand', 'greedy', 'human'],
                        default='rand', help='type of agent/model for baseline')
    parser.add_argument('--resblocks', type=int, default=4, help='number of basic blocks for resnets')

    # Hardware
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device for pytorch models')

    return parser.parse_args()


def sync_checkpoint(rank, comm: MPI.Intracomm, args, new_pi, old_pi):
    checkpath = os.path.join(args.savedir, 'checkpoint.pt')
    if rank == 0:
        torch.save(new_pi.pytorch_model.state_dict(), checkpath)
    comm.Barrier()
    # Update other policy
    old_pi.pytorch_model.load_state_dict(torch.load(checkpath))


def sync_data(rank, comm: MPI.Intracomm, args):
    if rank == 0:
        checkpath = os.path.join(args.savedir, 'checkpoint.pt')
        if args.checkpoint:
            assert os.path.exists(checkpath)
        else:
            # Clear worker data
            episodesdir = args.episodesdir
            data.clear_episodesdir(episodesdir)
            # Save new model
            new_model, _ = create_agent(args, '', load_checkpoint=False)

            if not os.path.exists(args.savedir):
                os.mkdir(args.savedir)
            torch.save(new_model.state_dict(), checkpath)
    parallel_err(rank, "Using checkpoint: {}".format(args.checkpoint))
    comm.Barrier()


def create_agent(args, name, use_base=False, load_checkpoint=True):
    agent = args.baseagent if use_base else args.agent
    if agent == 'mcts':
        model = value.ValueNet(args.boardsize, args.resblocks)
        pi = policies.MCTS(name, model, args.mcts, args.temp, args.tempsteps)
    elif agent == 'ac':
        model = actorcritic.ActorCriticNet(args.boardsize)
        pi = policies.ActorCritic(name, model)
    elif agent == 'mcts-ac':
        model = actorcritic.ActorCriticNet(args.boardsize)
        pi = policies.MCTSActorCritic(name, model, args.branches, args.depth)
    elif agent == 'rand':
        model = None
        pi = policies.RAND_PI
    elif agent == 'greedy':
        model = None
        pi = policies.GREEDY_PI
    elif agent == 'human':
        model = None
        pi = policies.HUMAN_PI
    else:
        raise Exception("Unknown agent argument", agent)

    if load_checkpoint and not use_base:
        check_path = os.path.join(args.savedir, 'checkpoint.pt')
        model.load_state_dict(torch.load(check_path))

    return model, pi
