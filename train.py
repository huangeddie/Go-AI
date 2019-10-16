import collections
import pickle
import random

import gym
from torch import multiprocessing as mp
from tqdm import tqdm

import hyperparameters
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *


def worker_train(rank, barrier, status, winrate):
    # Replay data
    replay_data = collections.deque(maxlen=REPLAY_MEMSIZE // WORKERS)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)
    if rank == 0:
        tqdm.write(f'{curr_model}\n')

    # Set parameters on disk
    if rank == 0:
        continue_checkpoint = hyperparameters.reset_disk_params(curr_model)
        tqdm.write(f"Continuing from checkpoint: {continue_checkpoint}\n")
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
            # Pit against checkpoint
            if rank == 0:
                winrate.value = 0
            barrier.wait()
            wr, _ = game.play_games(go_env, curr_pi, checkpoint_pi, False, NUM_EVALGAMES // WORKERS)
            with winrate.get_lock():
                winrate.value += (wr / WORKERS)
            barrier.wait()
            if rank == 0:
                tqdm.write(f"Winrate against checkpoint: {100 * winrate.value:.1f}%")
            barrier.wait()

            # Update checkpoint
            if winrate.value > 0.6:
                if rank == 0:
                    torch.save(curr_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                barrier.wait()

                # Update checkpoint policy
                checkpoint_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))

                if rank == 0:
                    tqdm.write("Accepted new model")

                    # Plot samples of states and response heatmaps
                    metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJPATH)
                    tqdm.write("Plotted sample trajectory")
                barrier.wait()

                # See how it fairs against the baselines
                for opponent in [policies.RAND_PI, policies.GREEDY_PI, policies.MCT_GREEDY_PI]:
                    if rank == 0:
                        winrate.value = 0
                    barrier.wait()

                    wr, _ = game.play_games(go_env, curr_pi, opponent, False, NUM_EVALGAMES // WORKERS)
                    with winrate.get_lock():
                        winrate.value += (wr / WORKERS)
                    barrier.wait()
                    if rank == 0:
                        tqdm.write(f"Win rate against {opponent}: {100 * winrate.value:.1f}%")

            elif winrate.value < 0.4:
                if rank == 0:
                    torch.save(checkpoint_pi.pytorch_model.state_dict(), CHECKPOINT_PATH)
                barrier.wait()
                curr_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))
                if rank == 0:
                    tqdm.write("Rejected new model")

            barrier.wait()

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        if rank == 0:
            tqdm.write(f"Temp decayed to {curr_pi.temp:.5f}, {checkpoint_pi.temp:.5f}\n")


if __name__ == '__main__':
    # Parallel Setup
    mp.set_start_method('spawn')
    barrier = mp.Barrier(WORKERS)
    status = mp.Value('d', 0.0)
    winrate = mp.Value('d', 0.0)

    tqdm.write(f'{WORKERS} Workers\n')

    if WORKERS <= 1:
        worker_train(0, barrier, status, winrate)
    else:
        mp.spawn(fn=worker_train, args=(barrier, status, winrate), nprocs=WORKERS, join=True)
