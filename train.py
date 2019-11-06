import collections
from datetime import datetime

import gym
import torch
from mpi4py import MPI

import utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models, actorcritic_model


def worker_train(rank: int, args, comm: MPI.Intracomm):
    # Replay data
    replay_data = collections.deque(maxlen=args.replaysize // comm.Get_size())

    # Set parameters and episode data on disk
    utils.sync_data(rank, comm, args)

    # Load replay data if any
    if args.checkpoint:
        replay_data = data.load_replaydata(args.episodes_dir, rank)
        utils.parallel_err(rank, 'Loaded replay data')

    # Policies and Model
    if args.agent == 'mcts':
        curr_model = value_models.ValueNet(args.boardsize)
        checkpoint_model = value_models.ValueNet(args.boardsize)
        curr_pi = policies.MCTS('Current', curr_model, args.mcts, args.temp, args.tempsteps)
        checkpoint_pi = policies.MCTS('Checkpoint', checkpoint_model, args.mcts, args.temp, args.tempsteps)
    elif args.agent == 'ac':
        curr_model = actorcritic_model.ActorCriticNet(args.boardsize)
        checkpoint_model = actorcritic_model.ActorCriticNet(args.boardsize)
        curr_pi = policies.ActorCritic('Current', curr_model)
        checkpoint_pi = policies.ActorCritic('Checkpoint', checkpoint_model)
    else:
        raise Exception("Unknown Agent Argument", args.agent)

    # Sync parameters from disk
    curr_model.load_state_dict(torch.load(args.check_path))
    checkpoint_model.load_state_dict(torch.load(args.check_path))
    optim = torch.optim.Adam(curr_model.parameters(), args.learning_rate) if rank == 0 else None

    # Environment
    go_env = gym.make('gym_go:go-v0', size=args.boardsize)

    # Header output
    utils.parallel_out(rank, "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tC_WR\tR_WR\tG_WR")

    # Training
    starttime = datetime.now()
    replay_len = 0
    check_winrate, rand_winrate, greedy_winrate = 0, 0, 0,
    crit_acc, crit_loss, act_acc, act_loss = 0, 0, 0, 0,
    for iteration in range(args.iterations):
        # Log a Sample Trajectory
        if rank == 0 and args.demotraj_path is not None:
            metrics.plot_traj_fig(go_env, checkpoint_pi, args.demotraj_path)
            utils.parallel_err(rank, "Plotted trajectory")

        # Play episodes
        wr, trajectories, avg_gametime = utils.parallel_play(comm, go_env, checkpoint_pi, checkpoint_pi, True,
                                                             args.episodes)
        utils.parallel_err(rank, f'{checkpoint_pi} V {checkpoint_pi} | {avg_gametime:.1f}S/GAME, {100 * wr:.1f}% WIN')
        replay_data.extend(trajectories)

        # Write episodes
        data.save_replaydata(rank, replay_data, args.episodes_dir)
        comm.Barrier()
        utils.parallel_err(rank, 'Wrote all replay data to disk')

        # Optimize
        if rank == 0:
            # Sample data as batches
            trainadata, replay_len = data.sample_replaydata(args.episodes_dir, args.trainstep_size, args.batchsize)

            # Optimize
            utils.parallel_err(rank, 'Optimizing on worker 0...')
            if args.agent == 'mcts':
                crit_acc, crit_loss = value_models.optimize(curr_model, trainadata, optim)
                act_acc, act_loss = 0, 0
            elif args.agent == 'ac':
                crit_acc, crit_loss, act_acc, act_loss = actorcritic_model.optimize(curr_model, trainadata, optim)

            torch.save(curr_model.state_dict(), args.tmp_path)
        comm.Barrier()

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(args.tmp_path))

        # Model Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            # See how this new model compares
            for opponent in [checkpoint_pi, policies.RAND_PI, policies.GREEDY_PI]:
                # Play some games
                wr, _, avg_gametime = utils.parallel_play(comm, go_env, curr_pi, opponent, False,
                                                                     args.evaluations)
                utils.parallel_err(rank, f'{curr_pi} V {opponent} | {avg_gametime:.1f}S/GAME, {100 * wr:.1f}% WIN')

                # Do stuff based on the opponent we faced
                if opponent == checkpoint_pi:
                    check_winrate = wr

                    # Sync checkpoint
                    if check_winrate > 0.55:
                        utils.sync_checkpoint(rank, comm, newcheckpoint_pi=curr_pi, check_path=args.check_path,
                                              other_pi=checkpoint_pi)
                        utils.parallel_err(rank, f"Accepted new checkpoint")

                        # Clear episodes
                        replay_data.clear()
                        utils.parallel_err(rank, "Cleared replay data")
                    elif check_winrate < 0.4:
                        utils.sync_checkpoint(rank, comm, newcheckpoint_pi=checkpoint_pi, check_path=args.check_path,
                                              other_pi=curr_pi)
                        # Break out of comparing to other models since we know it's bad
                        utils.parallel_err(rank, f"Rejected new checkpoint")
                        break
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
    world_size = int(comm.Get_size())

    utils.parallel_err(rank, f"{world_size} Workers, Board Size {args.boardsize}, Model '{args.agent}', "
                             f'Temp {args.temp:.4f}')

    worker_train(rank, args, comm)
