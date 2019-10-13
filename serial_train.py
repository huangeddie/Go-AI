import random
import collections

import gym
import torch

import train_utils
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
from hyperparameters import *

if __name__ == '__main__':
    # Model
    curr_model = value_models.ValueNet(BOARD_SIZE)
    checkpoint_model = value_models.ValueNet(BOARD_SIZE)
    print(curr_model)

    # Set parameters on disk
    train_utils.set_disk_params(LOAD_SAVED_MODELS, CHECKPOINT_PATH, TMP_PATH, curr_model)

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
    metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)

    # Replay data
    replay_data = collections.deque(maxlen=REPLAY_MEM_SIZE)

    # Training
    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration} | Replay size: {len(replay_data)}")

        # Make and write out the episode data
        _, trajectories = game.play_games(go_env, curr_pi, curr_pi, True, EPISODES_PER_ITERATION)
        replay_data.extend(trajectories)

        # Process the data
        train_data = random.sample(replay_data, min(DATA_SIZE_PER_ITERATION, len(replay_data)))
        train_data = data.replaylist_to_numpy(train_data)

        # Optimize
        value_models.optimize(curr_model, train_data, optim, BATCH_SIZE)

        # Evaluate
        status = train_utils.update_checkpoint(go_env, curr_pi, checkpoint_pi, NUM_EVAL_GAMES, CHECKPOINT_PATH)

        if status == 1:
            # Update checkpoint policy
            checkpoint_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))
            print("Accepted new model")

            # Plot samples of states and response heatmaps
            metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)
            print("Plotted sample trajectory")

            # See how it fairs against the baselines
            game.play_games(go_env, curr_pi, policies.RAND_PI, False, NUM_EVAL_GAMES)
            game.play_games(go_env, curr_pi, policies.GREEDY_PI, False, NUM_EVAL_GAMES)
            game.play_games(go_env, curr_pi, policies.MCT_GREEDY_PI, False, NUM_EVAL_GAMES)
        elif status == -1:
            curr_pi.pytorch_model.load_state_dict(torch.load(CHECKPOINT_PATH))
            print("Rejected new model")

        # Decay the temperatures if any
        curr_pi.decay_temp(TEMP_DECAY)
        checkpoint_pi.decay_temp(TEMP_DECAY)
        print(f"Temp decayed to {curr_pi.temp:.5f}, {checkpoint_pi.temp:.5f}")

    # Evaluate
    checkpoint_pi.set_temp(0)
    game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
