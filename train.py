import collections
import os
from datetime import datetime

import gym
import torch
from mpi4py import MPI

import go_ai.parallel
from go_ai import policies, data, utils
from go_ai.models import value, actorcritic


def train(comm, args, curr_pi, checkpoint_pi):
    # Environment
    go_env = gym.make('gym_go:go-v0', size=args.boardsize, reward_method=args.reward)

    # Optimizer
    curr_model = curr_pi.pytorch_model
    optim = torch.optim.Adam(curr_model.parameters(), args.lr, weight_decay=1e-4)

    # Replay data
    world_size = comm.Get_size()
    replay_data = collections.deque(maxlen=args.replaysize // world_size)

    # Training
    tmp_path = os.path.join(args.savedir, 'tmp.pt')
    rank = comm.Get_rank()
    starttime = datetime.now()
    winrates = collections.defaultdict(float)
    crit_acc, crit_loss, act_acc, act_loss = 0, 0, 0, 0,

    # Header output
    go_ai.parallel.parallel_out(comm, "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tC_WR\tR_WR\tG_WR")

    for iteration in range(args.iterations):
        # Play episodes
        go_ai.parallel.parallel_err(comm, f'Self-Playing {checkpoint_pi} V {checkpoint_pi}...')
        wr, trajectories = go_ai.parallel.parallel_play(comm, go_env, checkpoint_pi, checkpoint_pi, True, args.episodes)

        replay_data.extend(trajectories)

        # Write episodes
        data.save_replaydata(comm, replay_data, args.episodesdir)
        comm.Barrier()
        go_ai.parallel.parallel_err(comm, 'Wrote all replay data to disk')

        # Sample data as batches
        trainadata, replay_len = data.sample_replaydata(comm, args.episodesdir, args.trainsize, args.batchsize)

        # Optimize
        go_ai.parallel.parallel_err(comm, f'Optimizing in {len(trainadata)} training steps...')
        if args.agent == 'mcts':
            crit_acc, crit_loss = value.optimize(comm, curr_model, trainadata, optim)
            act_acc, act_loss = 0, 0
        elif args.agent == 'ac' or args.agent == 'mcts-ac':
            if rank == 0:
                crit_acc, crit_loss, act_acc, act_loss = actorcritic.optimize(comm, curr_model, trainadata, optim)

        # Sync model
        if rank == 0:
            torch.save(curr_model.state_dict(), tmp_path)
        comm.Barrier()

        go_ai.parallel.parallel_err(comm, f'Optimized | {crit_acc * 100 :.1f}% {crit_loss:.3f}L')

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(tmp_path))

        # Model Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            # See how this new model compares
            for opponent in [checkpoint_pi, policies.RAND_PI, policies.GREEDY_PI]:
                # Play some games
                go_ai.parallel.parallel_err(comm, f'Pitting {curr_pi} V {opponent}')
                wr, _ = go_ai.parallel.parallel_play(comm, go_env, curr_pi, opponent, False, args.evaluations)
                winrates[opponent] = wr

                # Do stuff based on the opponent we faced
                if opponent == checkpoint_pi:

                    # Sync checkpoint
                    if wr > 0.55:
                        utils.sync_checkpoint(comm, args, new_pi=curr_pi, old_pi=checkpoint_pi)
                        go_ai.parallel.parallel_err(comm, f"Accepted new checkpoint")

                        # Clear episodes
                        replay_data.clear()
                        go_ai.parallel.parallel_err(comm, "Cleared replay data")
                    else:
                        go_ai.parallel.parallel_err(comm, f"Continuing to train candidate checkpoint")
                        break

        # Print iteration summary
        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = f"{str(delta).split('.')[0]}\t{iteration:02d}\t{replay_len:07d}\t" \
                    f"{100 * crit_acc:04.1f}\t{crit_loss:04.3f}\t" \
                    f"{100 * act_acc:04.1f}\t{act_loss:04.3f}\t" \
                    f"{100 * winrates[checkpoint_pi]:04.1f}\t{100 * winrates[policies.RAND_PI]:04.1f}\t" \
                    f"{100 * winrates[policies.GREEDY_PI]:04.1f}"
        go_ai.parallel.parallel_out(comm, iter_info)

if __name__ == '__main__':
    # Arguments
    args = utils.hyperparameters()

    # Parallel Setup
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    go_ai.parallel.parallel_err(comm, f"{world_size} Workers, {args}")

    # Set parameters and episode data on disk
    utils.sync_data(comm, args)

    # Model and Policies
    curr_model, curr_pi = utils.create_agent(args, 'Current', load_checkpoint=True)
    checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint', load_checkpoint=True)

    # Device
    device = torch.device(args.device)
    curr_model.to(device)
    checkpoint_model.to(device)

    # Train
    train(comm, args, curr_pi, checkpoint_pi)