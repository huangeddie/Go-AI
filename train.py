import collections
import os
import random

import gym
import torch
from torch import multiprocessing as mp
from tqdm import tqdm

from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *


def worker_train(rank, barrier, winrate):
    # Replay data
    replay_data = collections.deque(maxlen=REPLAY_MEMSIZE // WORKERS)
    if CONTINUE_CHECKPOINT:
        old_data = data.load_replaydata(EPISODES_DIR, rank)
        replay_data.extend(old_data)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)
    if rank == 0:
        tqdm.write('{}\n'.format(curr_model))

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
        tqdm.write("Continuing from checkpoint: {}\n".format(CONTINUE_CHECKPOINT))
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
    for iteration in range(ITERATIONS):
        if rank == 0:
            tqdm.write("Iteration {} | Worker 0 Replay Size: {}".format(iteration, len(replay_data)))
        barrier.wait()

        # Log a Sample Trajectory
        if rank == 0:
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
            train_data = random.sample(all_data, min(TRAINSAMPLE_MEMSIZE, len(all_data)))
            train_data = data.replaylist_to_numpy(train_data)

            del all_data

            # Optimize
            value_models.optimize(curr_model, train_data, optim, BATCH_SIZE)

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
            if rank == 0:
                tqdm.write("Winrate against checkpoint: {:.1f}%".format(100 * winrate.value))
            barrier.wait()

            # Update checkpoint
            if winrate.value > 0.6:
                if rank == 0:
                    torch.save(curr_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                    tqdm.write("Accepted new model")
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
                    if rank == 0:
                        tqdm.write("Win rate against {}: {:.1f}%".format(opponent, 100 * winrate.value))

            elif winrate.value < 0.4:
                if rank == 0:
                    torch.save(checkpoint_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                    tqdm.write("Rejected new model")
                barrier.wait()

                curr_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))

            barrier.wait()

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        if rank == 0:
            tqdm.write("Temp decayed to {:.5f}, {:.5f}\n".format(curr_pi.temp, checkpoint_pi.temp))


if __name__ == '__main__':
    # Parallel Setup
    mp.set_start_method('spawn')
    barrier = mp.Barrier(WORKERS)
    winrate = mp.Value('d', 0.0)

    tqdm.write('{}/{} Workers\n'.format(WORKERS, mp.cpu_count()))

    if WORKERS <= 1:
        worker_train(0, barrier, winrate)
    else:
        mp.spawn(fn=worker_train, args=(barrier, winrate), nprocs=WORKERS, join=True)
