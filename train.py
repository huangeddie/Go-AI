import collections
import os
import random
import sys
from datetime import datetime

import gym
import torch
from torch import multiprocessing as mp
from tqdm import tqdm

from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *


def worker_print(rank, s):
    if rank == 0:
        print(s)


def worker_train(rank, barrier, winrate):
    # Replay data
    replay_data = collections.deque(maxlen=REPLAY_MEMSIZE // WORKERS)
    if CONTINUE_CHECKPOINT:
        old_data = data.load_replaydata(EPISODES_DIR, rank)
        replay_data.extend(old_data)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)

    # Set parameters and data on disk
    if rank == 0:
        if CONTINUE_CHECKPOINT:
            assert os.path.exists(CHECKPOINT_PATH)
        else:
            # Clear worker data
            episode_files = os.listdir(EPISODES_DIR)
            for item in episode_files:
                if item.endswith(".pickle"):
                    os.remove(os.path.join(EPISODES_DIR, item))
            # Set parameters
            torch.save(curr_model.state_dict(), CHECKPOINT_PATH)
        tqdm.write("Continuing from checkpoint: {}".format(CONTINUE_CHECKPOINT), file=sys.stderr)
    barrier.wait()

    # Load parameters from disk
    curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    optim = torch.optim.Adam(curr_model.parameters(), 1e-3) if rank == 0 else None

    # Policies
    curr_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)
    checkpoint_pi = policies.MctPolicy('Checkpoint', checkpoint_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)

    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Training
    starttime = datetime.now()
    replay_len = 0
    check_winrate, rand_winrate, greedy_winrate, mctgreedy_wr = 0, 0, 0, 0
    pred_acc, pred_loss = 0, 0
    worker_print(rank, "TIME\tITR\tREPLAY\tACCUR\tLOSS\tTEMP\tC_WR\tR_WR\tG_WR")
    for iteration in range(ITERATIONS):
        # Log a Sample Trajectory
        if rank == 0 and DEMO_TRAJPATH is not None:
            metrics.plot_traj_fig(go_env, checkpoint_pi, DEMO_TRAJPATH)

        # Play episodes
        _, trajectories = game.play_games(go_env, checkpoint_pi, checkpoint_pi, True, EPISODES_PER_ITER // WORKERS)
        replay_data.extend(trajectories)

        # Write episodes to disk
        data.save_replaydata(replay_data, EPISODES_DIR, rank)
        barrier.wait()

        # Process the data
        if rank == 0:
            # Gather all workers' data to sample from
            all_data = data.load_replaydata(EPISODES_DIR)
            replay_len = len(all_data)
            train_data = random.sample(all_data, min(TRAINSAMPLE_MEMSIZE, len(all_data)))
            train_data = data.replaylist_to_numpy(train_data)

            del all_data

            # Optimize
            pred_acc, pred_loss = value_models.optimize(curr_model, train_data, optim, BATCH_SIZE)

            torch.save(curr_model.state_dict(), TMP_PATH)
        barrier.wait()

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(TMP_PATH))

        # Evaluate
        if (iteration + 1) % ITERS_PER_EVAL == 0:
            # Pit against checkpoint
            if rank == 0:
                winrate.value = 0
            barrier.wait()
            wr, _ = game.play_games(go_env, curr_pi, checkpoint_pi, False, NUM_EVALGAMES // WORKERS)
            with winrate.get_lock():
                winrate.value += (wr / WORKERS)
            barrier.wait()
            check_winrate = winrate.value

            # Update checkpoint
            if check_winrate > 0.55:
                if rank == 0:
                    torch.save(curr_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                barrier.wait()

                # Update checkpoint policy
                checkpoint_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))

                # See how it fairs against the baselines
                for opponent in [policies.RAND_PI, policies.GREEDY_PI]:
                    if rank == 0:
                        winrate.value = 0
                    barrier.wait()

                    wr, _ = game.play_games(go_env, curr_pi, opponent, False, NUM_EVALGAMES // WORKERS)
                    with winrate.get_lock():
                        winrate.value += (wr / WORKERS)
                    barrier.wait()
                    if opponent == policies.RAND_PI:
                        rand_winrate = winrate.value
                    elif opponent == policies.GREEDY_PI:
                        greedy_winrate = winrate.value

            elif check_winrate < 0.4:
                if rank == 0:
                    torch.save(checkpoint_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                barrier.wait()

                curr_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))

        currtime = datetime.now()
        delta = currtime - starttime
        iter_info = "{}\t{}\t{:07d}\t{:.1f}\t{:.3f}\t{:.4f}".format(str(delta).split('.')[0], iteration, replay_len,
                                                                    100 * pred_acc, curr_pi.temp, pred_loss) \
                    + "\t{:.1f}\t{:.1f}\t{:.1f}".format(100 * check_winrate, 100 * rand_winrate, 100 * greedy_winrate)
        worker_print(rank, iter_info)

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)


if __name__ == '__main__':
    # Parallel Setup
    mp.set_start_method('spawn')
    barrier = mp.Barrier(WORKERS)
    winrate = mp.Value('d', 0.0)

    tqdm.write('{}/{} Workers'.format(WORKERS, mp.cpu_count()), file=sys.stderr)

    if WORKERS <= 1:
        worker_train(0, barrier, winrate)
    else:
        mp.spawn(fn=worker_train, args=(barrier, winrate), nprocs=WORKERS, join=True)
