import collections
import random
import sys
from datetime import datetime

import gym
import torch
from torch import multiprocessing as mp
from tqdm import tqdm

import utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models


def worker_train(rank, args, barrier, winrate):
    # Replay data
    replay_data = collections.deque(maxlen=args.replaysize // args.workers)
    if args.checkpoint:
        old_data = data.load_replaydata(args.episodes_dir, rank)
        replay_data.extend(old_data)

    # Set parameters and episode data on disk
    utils.sync_data(args, barrier, rank)

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
    utils.parallel_print(rank, "TIME\tITR\tREPLAY\tACCUR\tLOSS\tTEMP\tC_WR\tR_WR\tG_WR")
    for iteration in range(args.iterations):
        # Log a Sample Trajectory
        if rank == 0 and args.demotraj_path is not None:
            metrics.plot_traj_fig(go_env, checkpoint_pi, args.demotraj_path)

        # Play episodes
        _, trajectories = game.play_games(go_env, checkpoint_pi, checkpoint_pi, True, args.episodes // args.workers)
        replay_data.extend(trajectories)

        # Write episodes to disk
        data.save_replaydata(replay_data, args.episodes_dir, rank)
        barrier.wait()

        # Optimize
        if rank == 0:
            # Gather all workers' data to sample from
            all_data = data.load_replaydata(args.episodes_dir)
            replay_len = len(all_data)
            train_data = random.sample(all_data, min(args.trainstep_size, len(all_data)))
            train_data = data.replaylist_to_numpy(train_data)

            del all_data

            # Optimize
            pred_acc, pred_loss = value_models.optimize(curr_model, train_data, optim, args.batchsize)

            torch.save(curr_model.state_dict(), args.tmp_path)
        barrier.wait()

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(args.tmp_path))

        # Model Evaluation
        if (iteration + 1) % args.eval_interval == 0:
            # See how this new model compares
            for opponent in [checkpoint_pi, policies.RAND_PI, policies.GREEDY_PI]:
                # Reset winrate
                if rank == 0:
                    winrate.value = 0
                barrier.wait()

                # Play some games
                wr, _ = game.play_games(go_env, curr_pi, opponent, False, args.evaluations // args.workers)
                with winrate.get_lock():
                    winrate.value += (wr / args.workers)
                barrier.wait()

                # Do stuff based on the opponent we faced
                if opponent == checkpoint_pi:
                    check_winrate = winrate.value

                    # Sync checkpoint
                    if check_winrate > 0.55:
                        utils.sync_checkpoint(rank, barrier, newcheckpoint_pi=curr_pi, check_path=args.check_path,
                                              other_pi=checkpoint_pi)
                    elif check_winrate < 0.4:
                        utils.sync_checkpoint(rank, barrier, newcheckpoint_pi=checkpoint_pi, check_path=args.check_path,
                                              other_pi=curr_pi)
                        # Break out of comparing to other models since we know it's bad
                        break
                elif opponent == policies.RAND_PI:
                    rand_winrate = winrate.value
                elif opponent == policies.GREEDY_PI:
                    greedy_winrate = winrate.value

        # Print iteration summary
        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = "{}\t{}\t{:07d}\t{:.1f}\t{:.3f}\t{:.4f}".format(str(delta).split('.')[0], iteration, replay_len,
                                                                    100 * pred_acc, pred_loss, curr_pi.temp) \
                    + "\t{:.1f}\t{:.1f}\t{:.1f}".format(100 * check_winrate, 100 * rand_winrate, 100 * greedy_winrate)
        utils.parallel_print(rank, iter_info)

        # Decay the temperatures if any
        curr_pi.decay_temp(args.tempdecay)
        checkpoint_pi.decay_temp(args.tempdecay)


if __name__ == '__main__':
    # Arguments
    args = utils.hyperparameters()

    # Parallel Setup
    if args.spawnmethod is not None:
        mp.set_start_method(args.spawnmethod)
    tqdm.write('{}/{} Workers, Board Size {}, Spawn method: {}'.format(args.workers, mp.cpu_count(), args.boardsize,
                                                                       args.spawnmethod), file=sys.stderr)
    barrier = mp.Barrier(args.workers)
    winrate = mp.Value('d', 0.0)

    if args.workers <= 1:
        worker_train(0, args, barrier, winrate)
    else:
        procs = []
        for w in range(args.workers):
            p = mp.Process(target=worker_train, args=(w, args, barrier, winrate))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
