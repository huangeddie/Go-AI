import gym
from tqdm import tqdm

from go_ai import policies

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


def play_games(go_env, first_policy: policies.Policy, second_policy: policies.Policy, get_traj, episodes):
    replay_data = []
    wins = 0
    pbar = tqdm(range(1, episodes + 1), desc="{} vs. {}".format(first_policy.name, second_policy.name), position=0)
    for i in pbar:
        go_env.reset()
        if i % 2 == 0:
            w, traj = pit(go_env, first_policy, second_policy, get_traj=get_traj)
        else:
            w, traj = pit(go_env, second_policy, first_policy, get_traj=get_traj)
            w = 1 - w
        wins += w
        replay_data.extend(traj)
        pbar.set_postfix_str("{:.1f}%".format(100 * wins / i))

    return wins / episodes, replay_data


def pit(go_env, black_policy: policies.Policy, white_policy: policies.Policy, get_traj=False):
    """
    Pits two policies against each other and returns the results
    :param get_trajectory: Whether to store trajectory in memory
    :param go_env:
    :param black_policy:
    :param white_policy:
    :return: Whether or not black won {1, 0.5, 0}, trajectory
        trajectory is a list of events where each event is of the form
        (canonical_state, action, canonical_next_state, reward, terminal, win)

        Trajectory is empty list if get_trajectory is None
    """
    num_steps = 0
    state = go_env.get_state()

    # Sync policies to current state
    black_policy.reset(state)
    white_policy.reset(state)

    max_steps = 2 * (go_env.size ** 2)

    mem_cache = []

    done = False

    while not done:
        # Get turn
        curr_turn = go_env.turn()

        # Get canonical state for policy and memory
        canonical_state = GoGame.get_canonical_form(state, curr_turn)

        # Get an action
        if curr_turn == GoVars.BLACK:
            action_probs = black_policy(state, step=num_steps)
        else:
            assert curr_turn == GoVars.WHITE
            action_probs = white_policy(state, step=num_steps)

        action = GoGame.random_weighted_action(action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, _ = go_env.step(action)

        # Get canonical form of next state for memory
        canonical_next_state = GoGame.get_canonical_form(next_state, curr_turn)

        # Sync the policies
        black_policy.step(action)
        if white_policy != black_policy:
            white_policy.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        if get_traj:
            mem_cache.append((curr_turn, canonical_state, action, canonical_next_state, reward, done))

        # Increment steps
        num_steps += 1

        # Setup for next event
        state = next_state

    assert done

    # Determine who won
    black_won = go_env.get_winning()

    replay_mem = []
    for turn, canonical_state, action, canonical_next_state, reward, terminal in mem_cache:
        if turn == GoVars.BLACK:
            win = black_won
        else:
            win = 1 - black_won

        replay_mem.append((canonical_state, action, canonical_next_state, reward, terminal, win))

    return black_won, replay_mem
