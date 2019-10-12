import gym
from go_ai import policies, game, metrics, data
from go_ai.models import value_models
import os
import random
import torch

# Hyperparameters
from hyperparameters import *

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

optim = torch.optim.Adam(curr_model.parameters(), 1e-3)
print(curr_model)

# Policies
curr_pi = policies.MctPolicy('Current', BOARD_SIZE, curr_model, INIT_TEMP, NUM_SEARCHES)
checkpoint_pi = policies.MctPolicy('Checkpoint', BOARD_SIZE, checkpoint_model, INIT_TEMP, NUM_SEARCHES)

curr_pi = policies.QTempPolicy('Current', curr_model, INIT_TEMP)
checkpoint_pi = policies.QTempPolicy('Checkpoint', checkpoint_model, INIT_TEMP)

rand_pi = policies.RandomPolicy()
greedy_pi = policies.QTempPolicy('Greedy', policies.greedy_val_func, 0)
greedymct_pi = policies.MctPolicy('MCT', BOARD_SIZE, policies.greedy_val_func, 0, NUM_SEARCHES)

human_policy = policies.HumanPolicy()

def decay_temps(policies, temp_decay, min_temp):
    for policy in policies:
        assert hasattr(policy, 'temp')
        policy.temp *= temp_decay
        if policy.temp < min_temp:
            policy.temp = min_temp
        print(f"{policy.name} temp decayed to {policy.temp}")

def set_temps(policies, temp):
    for policy in policies:
        assert hasattr(policy, 'temp')
        policy.temp = temp
        print(f"{policy.name} temp set to {policy.temp}")

# Sample Trajectory
metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)

# Training
for iteration in range(ITERATIONS):
    print(f"Iteration {iteration}")

    # Make and write out the episode data
    _, replay_data = game.play_games(go_env, curr_pi, curr_pi, True, EPISODES_PER_ITERATION)

    # Process the data
    random.shuffle(replay_data)
    replay_data = data.replaylist_to_numpy(replay_data)

    # Optimize
    value_models.optimize(curr_model, replay_data, optim, BATCH_SIZE)

    # Evaluate against checkpoint model and other baselines
    opp_winrate, _ = game.play_games(go_env, curr_pi, checkpoint_pi, False, NUM_EVAL_GAMES)

    if opp_winrate > 0.6:
        # New parameters are significantly better. Accept it
        torch.save(curr_model.state_dict(), CHECKPOINT_PATH)
        checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"{100 * opp_winrate:.1f}% Accepted new model")

        # Plot samples of states and response heatmaps
        metrics.plot_traj_fig(go_env, curr_pi, DEMO_TRAJECTORY_PATH)
        print("Plotted sample trajectory")

        rand_winrate, _ = game.play_games(go_env, curr_pi, rand_pi, False, NUM_EVAL_GAMES)
        greed_winrate, _ = game.play_games(go_env, curr_pi, greedy_pi, False, NUM_EVAL_GAMES)

    elif opp_winrate >= 0.4:
        # Keep trying
        print(f"{100 * opp_winrate:.1f}% Continuing to train current weights")
    else:
        # New parameters are significantly worse. Reject it.
        curr_model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"{100 * opp_winrate:.1f}% Rejected new model")

    # Decay the temperatures if any
    decay_temps([curr_pi, checkpoint_pi], TEMP_DECAY, MIN_TEMP)

# Evaluate
set_temps([curr_pi, checkpoint_pi], 0)
game.pit(go_env, human_policy, checkpoint_pi, False)
