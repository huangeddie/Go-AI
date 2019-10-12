import os
import random
import tempfile

import gym
import torch
from torch import multiprocessing as mp

from evaluation import evaluate
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *


def train(rank, tmp_path):
    barrier = mp.Barrier(WORKERS)

    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    optim = torch.optim.Adam(curr_model.parameters(), 1e-3)

    # Policy
    curr_pi = policies.MctPolicy('Current', curr_model, NUM_SEARCHES, INIT_TEMP, MIN_TEMP)

    # Play some games
    _, replay_data = game.play_games(go_env, curr_pi, curr_pi, True, EPISODES_PER_ITERATION // WORKERS)

    # Process the data
    random.shuffle(replay_data)
    replay_data = data.replaylist_to_numpy(replay_data)

    # Take turns optimizing
    for r in range(WORKERS):
        if r == rank:
            # My turn to optimize
            value_models.optimize(curr_model, replay_data, optim, BATCH_SIZE)
            torch.save(curr_model.state_dict(), tmp_path)

        # Get new parameters
        barrier.wait()
        barrier.reset()
        curr_model.load_state_dict(torch.load(tmp_path))


if __name__ == '__main__':
    # Environment
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)

    if LOAD_SAVED_MODELS:
        assert os.path.exists(CHECKPOINT_PATH)
        print("Starting from checkpoint")
    else:
        torch.save(curr_model.state_dict(), CHECKPOINT_PATH)
        print("Initialized checkpoint")

    curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))

    print(curr_model)

    # Policies
    curr_pi = policies.MctPolicy('Current', curr_model, NUM_SEARCHES, INIT_TEMP, MIN_TEMP)
    checkpoint_pi = policies.MctPolicy('Checkpoint', checkpoint_model, NUM_SEARCHES, INIT_TEMP, MIN_TEMP)

    rand_pi = policies.RandomPolicy()
    greedy_pi = policies.MctPolicy('Greedy', policies.greedy_val_func, num_searches=0, temp=0)
    greedymct_pi = policies.MctPolicy('MCT', policies.greedy_val_func, NUM_SEARCHES, temp=0)

    human_policy = policies.HumanPolicy()

    # Sample Trajectory
    metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)

    # Training
    tmp_path = tempfile.gettempdir() + "/tmp.checkpoint"

    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration}")

        mp.spawn(fn=train, args=(tmp_path,), nprocs=WORKERS, join=True)

        # Evaluate
        curr_model.load_state_dict(torch.load(tmp_path))
        accepted = evaluate(go_env, curr_pi, checkpoint_pi, NUM_EVAL_GAMES, CHECKPOINT_PATH)

        if accepted == 1:
            # Plot samples of states and response heatmaps
            metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)
            print("Plotted sample trajectory")

            # See how it fairs against the baselines
            rand_winrate, _ = game.play_games(go_env, curr_pi, rand_pi, False, NUM_EVAL_GAMES)
            greed_winrate, _ = game.play_games(go_env, curr_pi, greedy_pi, False, NUM_EVAL_GAMES)

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        print("Temp decayed to {:.5f}, {:.5f}".format(curr_pi.temp, checkpoint_pi.temp))

    # Evaluate
    checkpoint_pi.set_temp(0)
    game.pit(go_env, human_policy, checkpoint_pi, False)
