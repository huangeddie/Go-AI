import random
import sys

import gym
import torch
from torch import multiprocessing as mp
from tqdm import tqdm

import hyperparameters
import train_utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *
import collections
import pickle
import os


def worker_train(rank, barrier, status):
    # Replay data
    replay_data = collections.deque(maxlen=REPLAY_MEMSIZE // WORKERS)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)
    if rank == 0:
        tqdm.write(str(curr_model))

    # Set parameters on disk
    if rank == 0:
        continue_checkpoint = hyperparameters.reset_disk_params(curr_model)
        tqdm.write(f"Continuing from checkpoint: {continue_checkpoint}")
    barrier.wait()

    # Load parameters from disk
    curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    optim = torch.optim.Adam(curr_model.parameters(), 1e-3)

    # Policies
    curr_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)
    checkpoint_pi = policies.MctPolicy('Checkpoint', checkpoint_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)

    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Log a Sample Trajectory
    if rank == 0:
        metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJPATH)

    # Training
    for iteration in range(ITERATIONS):
        if rank == 0:
            tqdm.write(f"Iteration {iteration} | Replay size: {len(replay_data)}")
        barrier.wait()

        # Make and write out the episode data
        _, trajectories = game.play_games(go_env, curr_pi, curr_pi, True, EPISODES_PER_ITER // WORKERS)
        replay_data.extend(trajectories)

        with open(f"{EPISODES_DIR}worker_{rank}.pickle", 'wb') as f:
            pickle.dump(replay_data, f)

        barrier.wait()

        # Process the data
        if rank == 0:
            # Gather all workers' data to sample from
            all_data = []
            files = os.listdir(EPISODES_DIR)
            for file in files:
                if '.pickle' in file:
                    with open(EPISODES_DIR + file, 'rb') as f:
                        worker_data = pickle.load(f)
                        all_data.extend(worker_data)
            train_data = random.sample(all_data, min(TRAINSTEP_MEMSIZE, len(all_data)))
            train_data = data.replaylist_to_numpy(train_data)

            # Optimize
            value_models.optimize(curr_model, train_data, optim, BATCH_SIZE)

            torch.save(curr_model.state_dict(), TMP_PATH)
        barrier.wait()

        # Update model from worker 0's optimization
        curr_model.load_state_dict(torch.load(TMP_PATH))

        # Evaluate
        if (iteration + 1) % ITERS_PER_EVAL == 0:
            if rank == 0:
                status.value = update_checkpoint(go_env, curr_pi, checkpoint_pi, NUM_EVALGAMES,
                                                 CHECKPOINT_PATH)
            barrier.wait()

            if status.value == 1:
                # Update checkpoint policy
                checkpoint_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))

                if rank == 0:
                    tqdm.write("Accepted new model")

                    # Plot samples of states and response heatmaps
                    metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJPATH)
                    tqdm.write("Plotted sample trajectory")

                    # See how it fairs against the baselines
                    game.play_games(go_env, curr_pi, policies.RAND_PI, False, NUM_EVALGAMES)
                    game.play_games(go_env, curr_pi, policies.GREEDY_PI, False, NUM_EVALGAMES)
                    game.play_games(go_env, curr_pi, policies.MCT_GREEDY_PI, False, NUM_EVALGAMES)
            elif status.value == -1:
                curr_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))
                if rank == 0:
                    tqdm.write("Rejected new model")

            barrier.wait()

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        if rank == 0:
            tqdm.write(f"Temp decayed to {curr_pi.temp:.5f}, {checkpoint_pi.temp:.5f}")

def update_checkpoint(go_env, first_pi: policies.Policy, second_pi: policies.Policy, num_games, checkpoint_path):
    """
    Writes the PyTorch model parameters of the best policy to the checkpoint
    :param go_env:
    :param first_pi:
    :param second_pi:
    :param num_games:
    :param checkpoint_path:
    :return:
    * 1 = if first policy was better and its parameters were written to checkpoint
    * 0 = no policy was significantly better than the other, so nothing was written
    * -1 = the second policy was better and its parameters were written to checkpoint
    """
    # Evaluate against checkpoint model and other baselines
    first_winrate, _ = game.play_games(go_env, first_pi, second_pi, False, num_games)

    # Get the pytorch models
    first_model = first_pi.pytorch_model
    second_model = second_pi.pytorch_model
    assert isinstance(first_model, torch.nn.Module)
    assert isinstance(second_model, torch.nn.Module)

    if first_winrate > 0.6:
        # First policy was best
        torch.save(first_model.state_dict(), checkpoint_path)
        return 1
    elif first_winrate >= 0.4:
        return 0
    else:
        assert first_winrate < 0.4
        # Second policy was best
        torch.save(second_model.state_dict(), checkpoint_path)
        return -1


if __name__ == '__main__':
    # Parallel Setup
    mp.set_start_method('spawn')
    barrier = mp.Barrier(WORKERS)
    status = mp.Value('d', 0.0)

    tqdm.write(f'{WORKERS} Workers')

    if WORKERS <= 1:
        worker_train(0, barrier, status)
    else:
        mp.spawn(fn=worker_train, args=(barrier, status), nprocs=WORKERS, join=True)