import collections
from datetime import datetime

import gym
import torch
from mpi4py import MPI

import utils
from go_ai import policies, data
from go_ai.models import value, actorcritic


def worker_train(args, comm: MPI.Intracomm):
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Replay data
    replay_data = collections.deque(maxlen=args.replaysize // world_size)

    # Set parameters and episode data on disk
    utils.sync_data(rank, comm, args)

    # Policies and Model
    if args.agent == 'mcts':
        curr_model = value.ValueNet(args.boardsize)
        checkpoint_model = value.ValueNet(args.boardsize)
        curr_pi = policies.MCTS('Current', curr_model, args.mcts, args.temp, args.tempsteps)
        checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.temp, args.tempsteps)
    elif args.agent == 'ac':
        curr_model = actorcritic.ActorCriticNet(args.boardsize)
        checkpoint_model = actorcritic.ActorCriticNet(args.boardsize)
        curr_pi = policies.ActorCritic('Current', curr_model)
        checkpoint_pi = policies.ActorCritic('Checkpoint', checkpoint_model)
    else:
        raise Exception("Unknown Agent Argument", args.agent)

    # Sync parameters from disk
    curr_model.load_state_dict(torch.load(args.checkpath))
    checkpoint_model.load_state_dict(torch.load(args.checkpath))
    optim = torch.optim.Adam(curr_model.parameters(), args.lr)

    # Environment
    go_env = gym.make('gym_go:go-v0', size=args.boardsize)

    # Header output
    utils.parallel_out(rank, "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tC_WR\tR_WR\tG_WR")

    # Training
    starttime = datetime.now()
    check_winrate, rand_winrate, greedy_winrate = 0, 0, 0,
    crit_acc, crit_loss, act_acc, act_loss = 0, 0, 0, 0,
    for iteration in range(args.iterations):
        # Play episodes
        utils.parallel_err(rank, f'Self-Playing {checkpoint_pi} V {checkpoint_pi}')
        wr, trajectories = utils.parallel_play(comm, go_env, checkpoint_pi, checkpoint_pi, True,
                                                             args.episodes)

        replay_data.extend(trajectories)

        # Write episodes
        data.save_replaydata(rank, replay_data, args.episodesdir)
        comm.Barrier()
        utils.parallel_err(rank, 'Wrote all replay data to disk')

        # Sample data as batches
        trainadata, replay_len = data.sample_replaydata(args.episodesdir, args.trainsize // world_size, args.batchsize)

        # Optimize
        utils.parallel_err(rank, f'Optimizing in {len(trainadata)} training steps...')
        if args.agent == 'mcts':
            crit_acc, crit_loss = value.optimize(comm, curr_model, trainadata, optim)
            act_acc, act_loss = 0, 0
        elif args.agent == 'ac':
            if rank == 0:
                crit_acc, crit_loss, act_acc, act_loss = actorcritic.optimize(comm, curr_model, trainadata, optim)

        # Sync model
        if rank == 0:
            torch.save(curr_model.state_dict(), args.tmppath)
        comm.Barrier()

        utils.parallel_err(rank, f'Optimized | {crit_acc * 100 :.1f}% {crit_loss:.3f}L')

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(args.tmppath))

        # Model Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            # See how this new model compares
            for opponent in [checkpoint_pi, policies.RAND_PI, policies.GREEDY_PI]:
                # Play some games
                utils.parallel_err(rank, f'Pitting {curr_pi} V {opponent}')
                wr, _ = utils.parallel_play(comm, go_env, curr_pi, opponent, False,
                                                          args.evaluations)

                # Do stuff based on the opponent we faced
                if opponent == checkpoint_pi:
                    check_winrate = wr

                    # Sync checkpoint
                    if check_winrate > 0.55:
                        utils.sync_checkpoint(rank, comm, newcheckpoint_pi=curr_pi, checkpath=args.checkpath,
                                              other_pi=checkpoint_pi)
                        utils.parallel_err(rank, f"Accepted new checkpoint")

                        # Clear episodes
                        replay_data.clear()
                        utils.parallel_err(rank, "Cleared replay data")
                    else:
                        utils.parallel_err(rank, f"Continuing to train candidate checkpoint")
                        break
                elif opponent == policies.RAND_PI:
                    rand_winrate = wr
                elif opponent == policies.GREEDY_PI:
                    greedy_winrate = wr

        # Print iteration summary
        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = f"{str(delta).split('.')[0]}\t{iteration:02d}\t{replay_len:07d}\t" \
                    f"{100 * crit_acc:04.1f}\t{crit_loss:04.3f}\t" \
                    f"{100 * act_acc:04.1f}\t{act_loss:04.3f}\t" \
                    f"{100 * check_winrate:04.1f}\t{100 * rand_winrate:04.1f}\t{100 * greedy_winrate:04.1f}"
        utils.parallel_out(rank, iter_info)


if __name__ == '__main__':
    # Arguments
    args = utils.hyperparameters()

    # Parallel Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    utils.parallel_err(rank, f"{world_size} Workers, {args}")

    worker_train(args, comm)
