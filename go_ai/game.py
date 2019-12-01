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
    wins = 0
    if progress:
        pbar = tqdm(range(1, episodes + 1), desc="{} vs. {}".format(first_policy, second_policy), leave=True)
    else:
        pbar = range(1, episodes + 1)
    for i in pbar:
        go_env.reset()
        if i % 2 == 0:
            w, traj = pit(go_env, first_policy, second_policy, get_traj=get_traj)
        else:
            w, traj = pit(go_env, second_policy, first_policy, get_traj=get_traj)
            w = -w
        wins += int(w == 1)
        replay_data.extend(traj)
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str("{:.1f}%".format(100 * wins / i))

    return wins / episodes, replay_data


def pit(go_env, black_policy: policies.Policy, white_policy: policies.Policy, get_traj=False):
    """
    Pits two policies against each other and returns the results
    :param get_trajectory: Whether to store trajectory in memory
    :param go_env:
    :param black_policy:
    :param white_policy:
    :return: Whether or not black won {1, 0, -1}, trajectory
        trajectory is a list of events where each event is of the form
        (canonical_state, action, canonical_next_state, reward, terminal, win)

        Trajectory is empty list if get_trajectory is None
    """
    num_steps = 0
    state = go_env.get_state()

    max_steps = 2 * (go_env.size ** 2)

    blackcache = []
    whitecache = []

    done = False

    while not go_env.game_ended():
        # Get turn
        curr_turn = go_env.turn()

        # Get canonical state for policy and memory
        canonical_state = GoGame.get_canonical_form(state)

        # Get an action
        if curr_turn == GoVars.BLACK:
            action_probs = black_policy(go_env, step=num_steps)
            cache = blackcache
        else:
            assert curr_turn == GoVars.WHITE
            action_probs = white_policy(go_env, step=num_steps)
            cache = whitecache

        action = GoGame.random_weighted_action(action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, _ = go_env.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        if get_traj:
            cache.append((canonical_state, action, reward, done))

        # Increment steps
        num_steps += 1

        # Setup for next event
        state = next_state

    assert done

    # Determine who won
    black_won = go_env.get_winning()

    replay_mems = []
    for black, cache in zip([1, -1], [blackcache, whitecache]):
        win = black * black_won
        mem = []
        for i, (canonical_state, action, reward, terminal) in enumerate(cache):
            if i < len(cache) - 1:
                canonical_nextstate = cache[i + 1][0]
            else:
                canonical_nextstate = np.zeros(canonical_state.shape)
            reward = reward * (1 - terminal) + win * terminal
            mem.append((canonical_state, action, canonical_nextstate, reward, terminal, win))
        replay_mems.append(mem)

    black_mem, white_mem = replay_mems[0], replay_mems[1]
    replay_mem = []
    assert len(black_mem) == len(white_mem) or len(black_mem) == 1 + len(white_mem), (len(black_mem), len(white_mem))
    for black_event, white_event in zip(black_mem, white_mem):
        replay_mem.append(black_event)
        replay_mem.append(white_event)
    if len(black_mem) == 1 + len(white_mem):
        replay_mem.append(black_mem[-1])

    return black_won, replay_mem
