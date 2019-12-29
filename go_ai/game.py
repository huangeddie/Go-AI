import gym
from tqdm import tqdm

from go_ai import policies

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.pis = []

    def get_events(self):
        events = []
        black_won = self.get_winner()
        n = len(self)
        zipped = zip(self.states, self.actions, self.rewards, self.next_states, self.pis)
        for i, (state, action, reward, next_state, pi) in enumerate(zipped):
            turn = i % 2
            if turn == 0:
                won = black_won
            else:
                won = -black_won

            terminal = i == n - 1

            events.append((state, action, reward, next_state, terminal, won, pi))

        return events

    def add_event(self, state, action, reward, next_state, pi):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.pis.append(pi)

    def set_win(self, black_won):
        self.rewards[-1] = black_won

    def get_winner(self):
        return self.rewards[-1]

    def __len__(self):
        n = len(self.states)
        assert len(self.actions) == n
        assert len(self.rewards) == n
        assert len(self.next_states) == n
        assert len(self.pis) == n

        return n


def pit(go_env, black_policy: policies.Policy, white_policy: policies.Policy):
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

    traj = Trajectory()

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
        traj.add_event(can_state, action, reward, next_state, pi)

        # Increment steps
        num_steps += 1

        # Setup for next event
        state = next_state

    assert done

    # Determine who won
    black_won = go_env.get_winning()

    traj.set_win(black_won)

    return black_won, num_steps, traj


def play_games(go_env, first_policy: policies.Policy, second_policy: policies.Policy, episodes, progress=True):
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
            black_won, steps, traj = pit(go_env, first_policy, second_policy)
            first_won = black_won
        else:
            black_won, steps, traj = pit(go_env, second_policy, first_policy)
            first_won = -black_won
        black_wins += int(black_won == 1)
        first_wins += int(first_won == 1)
        all_steps.append(steps)
        replay_data.append(traj)
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str("{:.1f}% WIN".format(100 * first_wins / i))

    return first_wins / episodes, black_wins / episodes, all_steps, replay_data
