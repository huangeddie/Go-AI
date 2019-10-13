import random
import shutil

import gym
import torch
from torch import multiprocessing as mp

import train_utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *


def train(rank, barrier, workers, temp, model_path):
    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    curr_model.load_state_dict(torch.load(model_path))
    optim = torch.optim.Adam(curr_model.parameters(), 1e-3)

    # Policy
    curr_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, temp)

    # Play some games
    _, replay_data = game.play_games(go_env, curr_pi, curr_pi, True, EPISODES_PER_ITERATION // workers)

    # Process the data
    random.shuffle(replay_data)
    replay_data = data.replaylist_to_numpy(replay_data)

    # Take turns optimizing
    for r in range(workers):
        if r == rank:
            # My turn to optimize
            value_models.optimize(curr_model, replay_data, optim, BATCH_SIZE)
            torch.save(curr_model.state_dict(), model_path)

        # Get new parameters
        barrier.wait()
        curr_model.load_state_dict(torch.load(model_path))


def asdf(rank, queue: mp.queue.Queue, workers, temp, curr_path, checkpoint_path):
    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Models
    curr_model = value_models.ValueNet(BOARD_SIZE)
    curr_model.load_state_dict(torch.load(curr_path))

    checkpoint_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model.load_state_dict(torch.load(checkpoint_path))

    # Policies
    curr_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, temp)
    checkpoint_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, temp)

    # Play some games
    winrate, _ = game.play_games(go_env, curr_pi, checkpoint_pi, False, NUM_EVAL_GAMES // workers)
    queue.put(winrate)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    barrier = mp.Barrier(WORKERS)
    win_queue = mp.Queue()

    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)

    train_utils.set_disk_params(LOAD_SAVED_MODELS, CHECKPOINT_PATH, TMP_PATH, curr_model)

    curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))

    print(curr_model)

    # Policies
    curr_pi = policies.MctPolicy('Current', curr_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)
    checkpoint_pi = policies.MctPolicy('Checkpoint', checkpoint_model, MCT_SEARCHES, INIT_TEMP, MIN_TEMP)

    # Sample Trajectory
    metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)

    # Training
    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration}")

        mp.spawn(fn=train, args=(barrier, WORKERS, curr_pi.temp, TMP_PATH), nprocs=WORKERS, join=True)

        # Evaluate
        curr_model.load_state_dict(torch.load(TMP_PATH))
        assert curr_pi.pytorch_model == curr_model
        status = train_utils.update_checkpoint(go_env, curr_pi, checkpoint_pi, NUM_EVAL_GAMES, CHECKPOINT_PATH)

        if status == 1:
            # Plot samples of states and response heatmaps
            metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)
            print("Plotted sample trajectory")

            # See how it fairs against the baselines
            game.play_games(go_env, curr_pi, policies.RAND_PI, False, NUM_EVAL_GAMES)
            game.play_games(go_env, curr_pi, policies.GREEDY_PI, False, NUM_EVAL_GAMES)
            game.play_games(go_env, curr_pi, policies.MCT_GREEDY_PI, False, NUM_EVAL_GAMES)
        elif status == -1:
            shutil.copy(CHECKPOINT_PATH, TMP_PATH)

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        print("Temp decayed to {:.5f}, {:.5f}".format(curr_pi.temp, checkpoint_pi.temp))

    # Evaluate
    checkpoint_pi.set_temp(0)
    game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
