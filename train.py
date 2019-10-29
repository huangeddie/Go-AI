import collections
import random
from datetime import datetime

import gym
import torch
from mpi4py import MPI

import utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models


def worker_train(rank: int, args, comm: MPI.Intracomm):
    # Replay data
    replay_data = collections.deque(maxlen=args.replaysize // comm.Get_size())
    if args.checkpoint:
        old_data = data.load_replaydata(args.episodes_dir, rank)
        replay_data.extend(old_data)

    # Set parameters and episode data on disk
    utils.sync_data(rank, comm, args)

    # Model
    curr_model = value_models.ValueNet(args.boardsize)
    checkpoint_model = value_models.ValueNet(args.boardsize)

    # Load parameters from disk
    curr_model.load_state_dict(torch.load(args.check_path))
    checkpoint_model.load_state_dict(torch.load(args.check_path))
    optim = torch.optim.Adam(curr_model.parameters(), 1e-3) if rank == 0 else None

    # Policies
    curr_pi = policies.MCTS('Current', curr_model, args.mcts, args.starttemp, args.mintemp)
    checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.starttemp, args.mintemp)

    # Environment
    go_env = gym.make('gym_go:go-v0', size=args.boardsize)

    # Training
    starttime = datetime.now()
    replay_len = 0
    check_winrate, rand_winrate, greedy_winrate, mctgreedy_wr = 0, 0, 0, 0
    pred_acc, pred_loss = 0, 0
    utils.parallel_out(rank, "TIME\tITR\tREPLAY\tACCUR\tLOSS\tTEMP\tC_WR\tR_WR\tG_WR")
    for iteration in range(args.iterations):
        # Log a Sample Trajectory
        if rank == 0 and args.demotraj_path is not None:
            metrics.plot_traj_fig(go_env, checkpoint_pi, args.demotraj_path)

        # Play episodes
        _, trajectories = game.play_games(go_env, checkpoint_pi, checkpoint_pi, True, args.episodes // comm.Get_size())
        replay_data.extend(trajectories)

        # Write episodes to disk
        data.save_replaydata(replay_data, args.episodes_dir, rank)
        comm.allgather(None)

        # Optimize
        if rank == 0:
            # Gather all workers' data to sample from
            all_data = data.load_replaydata(args.episodes_dir)
            replay_len = len(all_data)
            train_data = random.sample(all_data, min(args.trainstep_size, len(all_data)))
            train_data = data.replaylist_to_numpy(train_data)

            del all_data

            # Optimize
            pred_acc, pred_loss = value_models.optimmodulize(curr_model, train_data, optim, args.batchsize)

            torch.save(curr_model.state_dict(), args.tmp_path)
        comm.allgather(None)

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(args.tmp_path))

        # Model Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            # See how this new model compares
            for opponent in [checkpoint_pi, policies.RAND_PI, policies.GREEDY_PI]:
                # Play some games
                wr, _ = game.play_games(go_env, curr_pi, opponent, False, args.evaluations // comm.Get_size())
                wr = comm.allreduce(wr, op=MPI.SUM) / comm.Get_size()

                # Do stuff based on the opponent we faced
                if opponent == checkpoint_pi:
                    check_winrate = wr

                    # Sync checkpoint
                    if check_winrate > 0.55:
                        utils.sync_checkpoint(rank, comm, newcheckpoint_pi=curr_pi, check_path=args.check_path,
                                              other_pi=checkpoint_pi)
                        utils.parallel_err(rank, "Accepted new checkpoint")
                    elif check_winrate < 0.4:
                        utils.sync_checkpoint(rank, comm, newcheckpoint_pi=checkpoint_pi, check_path=args.check_path,
                                              other_pi=curr_pi)
                        # Break out of comparing to other models since we know it's bad
                        utils.parallel_err(rank, "Rejected new checkpoint")
                        break
                    else:
                        utils.parallel_err(rank, "Continuing to train candidate checkpoint")
                elif opponent == policies.RAND_PI:
                    rand_winrate = wr
                elif opponent == policies.GREEDY_PI:
                    greedy_winrate = wr

        # Print iteration summary
        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = "{}\t{}\t{:07d}\t{:.1f}\t{:.3f}\t{:.4f}".format(str(delta).split('.')[0], iteration, replay_len,
                                                                    100 * pred_acc, pred_loss, curr_pi.temp) \
                    + "\t{:.1f}\t{:.1f}\t{:.1f}".format(100 * check_winrate, 100 * rand_winrate, 100 * greedy_winrate)
        utils.parallel_out(rank, iter_info)

        # Decay the temperatures if any
        curr_pi.decay_temp(args.tempdecay)
        checkpoint_pi.decay_temp(args.tempdecay)


if __name__ == '__main__':
    # Arguments
    args = utils.hyperparameters()

    # Parallel Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = int(comm.Get_size())

    utils.parallel_err(rank, '{} Workers, Board Size {}')

    worker_train(rank, args, comm)
