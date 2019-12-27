import gym
import numpy as np
from tqdm import tqdm

from go_ai import policies

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


def play_games(go_env, first_policy: policies.Policy, second_policy: policies.Policy, get_traj, episodes,
               progress=True):
    replay_data = []
    all_steps = []
    first_wins = 0
    black_wins = 0
    if progress:
        pbar = tqdm(range(1, episodes + 1), desc="{} vs. {}".format(first_policy, second_policy), leave=True)
    else:
        pbar = range(1, episodes + 1)
    for i in pbar:
        go_env.reset()
        if i % 2 == 0:
            black_won, steps, traj = pit(go_env, first_policy, second_policy, get_traj=get_traj)
            first_won = black_won
        else:
            black_won, steps, traj = pit(go_env, second_policy, first_policy, get_traj=get_traj)
            first_won = -black_won
        black_wins += int(black_won == 1)
        first_wins += int(first_won == 1)
        all_steps.append(steps)
        replay_data.extend(traj)
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str("{:.1f}% WIN".format(100 * first_wins / i))

    return first_wins / episodes, black_wins / episodes, all_steps, replay_data


def pit(go_env, black_policy: policies.Policy, white_policy: policies.Policy, get_traj=False):
    """
    Pits two policies against each other and returns the results
    :param get_trajectory: Whether to store trajectory in memory
    :param go_env:
    :param black_policy:
    :param white_policy:
    :return:
        • Whether or not black won {1, 0, -1}
        • Number of steps
        • Trajectory
            - Trajectory is a list of events where each event is of the form
            (canonical_state, action, canonical_next_state, reward, terminal, win)

            Trajectory is empty list if get_trajectory is None
    """
    num_steps = 0
    state = go_env.get_state()

    max_steps = 2 * (go_env.size ** 2)

    cache = []

    done = False

    while not done:
        # Get turn
        curr_turn = go_env.turn()

        # Get canonical state for policy and memory
        can_state = GoGame.get_canonical_form(state)

        # Get an action
        if curr_turn == GoVars.BLACK:
            pi = black_policy(go_env, step=num_steps)
        else:
            assert curr_turn == GoVars.WHITE
            pi = white_policy(go_env, step=num_steps)

        action = GoGame.random_weighted_action(pi)

        # Execute actions in environment and MCT tree
        next_state, reward, done, _ = go_env.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        if get_traj:
            cache.append((curr_turn, can_state, action, reward, done, pi))

        # Increment steps
        num_steps += 1

        # Setup for next event
        state = next_state

    assert done

    # Determine who won
    black_won = go_env.get_winning()

    replay_mem = []

    for i, (curr_turn, can_state, action, reward, done, pi) in enumerate(cache):
        win = black_won if curr_turn == GoVars.BLACK else - black_won
        if i < len(cache) - 1:
            can_nextstate = cache[i + 1][1]
        else:
            can_nextstate = np.zeros(can_state.shape)
        reward = reward * (1 - done) + win * done
        replay_mem.append((can_state, action, reward, can_nextstate, done, win, pi))

    return black_won, num_steps, replay_mem
