import gym
import numpy as np
import torch

from go_ai import montecarlo
from go_ai.montecarlo import temperate_pi

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def greedy_val_func(states):
    if len(states) <= 0:
        return np.array([])
    board_area = GoGame.get_action_size(states[0]) - 1

    vals = []
    for state in states:
        black_area, white_area = GoGame.get_areas(state)
        if GoGame.get_game_ended(state):
            if black_area > white_area:
                val = 1
            elif black_area < white_area:
                val = -1
            else:
                val = 0
        else:
            val = (black_area - white_area) / (board_area)
        vals.append(val)
    vals = np.array(vals, dtype=np.float)
    return vals[:, np.newaxis]


def smart_greedy_val_func(states):
    if len(states) <= 0:
        return np.array([])
    board_area = GoGame.get_action_size(states[0]) - 1

    vals = []
    for state in states:
        black_area, white_area = GoGame.get_areas(state)
        blacklibs, whitelibs = GoGame.get_num_liberties(state)
        if GoGame.get_game_ended(state):
            if black_area > white_area:
                val = 1
            elif black_area < white_area:
                val = -1
            else:
                val = 0
        else:
            val = (black_area + blacklibs - white_area - whitelibs) / (2 * board_area)
        vals.append(val)
    vals = np.array(vals, dtype=np.float)
    return vals[:, np.newaxis]


def pytorch_to_numpy(model):
    """
    Automatically turns terminal states into 1, 0, -1 based on win status
    :param model:
    :return: The numpy equivalent of the pytorch value model
    """

    def val_func(states):
        """
        :param states: Numpy batch of states
        :return:
        """
        dtype = next(model.parameters()).type()
        model.eval()
        with torch.no_grad():
            states = torch.from_numpy(states).type(dtype)
            state_vals = model(states)
            vals = state_vals.detach().cpu().numpy()

        # Check for terminals
        for i, state in enumerate(states):
            if GoGame.get_game_ended(state):
                vals[i] = GoGame.get_winning(state)

        return vals

    return val_func


class Policy:
    """
    Interface for all types of policies
    """

    def __init__(self, name, temp=None, temp_steps=None):
        self.name = name
        self.temp = temp
        self.temp_steps = temp_steps
        self.pytorch_model = None

    def __call__(self, go_env, step=None):
        """
        :param go_env: Go environment
        :param step: the number of steps taken in the game so far
        :return: Action probabilities
        """
        pass

    def decay_temp(self, decay):
        self.temp *= decay
        if self.temp < 0:
            self.temp = 0

    def set_temp(self, temp):
        self.temp = temp

    def __str__(self):
        return "{} {}".format(self.__class__.__name__, self.name)


class Random(Policy):
    def __init__(self):
        super(Random, self).__init__('Random')

    def __call__(self, go_env, step=None):
        """
        :param go_env:
        :param step:
        :return: Action probabilities
        """

        valid_moves = go_env.get_valid_moves()
        return valid_moves / np.sum(valid_moves)


class Human(Policy):
    def __init__(self):
        super(Human, self).__init__('Human')

    def __call__(self, go_env, step=None):
        """
        :param go_env:
        :param step:
        :return: Action probabilities
        """
        valid_moves = go_env.get_valid_moves()

        # Human interface
        while True:
            player_action = go_env.render('human')
            state = go_env.get_state()
            player_action = GoGame.action_2d_to_1d(player_action, state)
            if valid_moves[player_action] > 0:
                break

        action_probs = np.zeros(GoGame.get_action_size(state))
        action_probs[player_action] = 1

        return action_probs


class MCTS(Policy):
    def __init__(self, name, val_func, num_searches, temp=0, temp_steps=0):
        super(MCTS, self).__init__(name, temp, temp_steps)
        if isinstance(val_func, torch.nn.Module):
            self.pytorch_model = val_func
            val_func = pytorch_to_numpy(val_func)

        self.val_func = val_func
        self.num_searches = num_searches

    def __call__(self, go_env, step=None):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """

        canonical_children, child_groupmaps = go_env.get_canonical_children()
        child_vals = self.val_func(np.array(canonical_children))
        canonical_state = go_env.get_canonical_state()

        valid_indicators = go_env.get_valid_moves()
        qvals = montecarlo.vals_to_qs(child_vals, canonical_state)

        # Search on grandchildren layer
        if self.num_searches > 0:
            valid_moves = np.argwhere(valid_indicators).flatten()
            assert len(valid_moves) == len(child_vals)
            for i, child_idx in enumerate(np.argsort(-child_vals.flatten())):
                action_to_child = valid_moves[child_idx]
                if i < self.num_searches:
                    child = canonical_children[child_idx]
                    if GoGame.get_game_ended(child):
                        continue
                    child_groupmap = child_groupmaps[child_idx]
                    grandchildren, _ = GoGame.get_canonical_children(child, child_groupmap)
                    grand_vals = self.val_func(np.array(grandchildren))
                    # Assume opponent would take action that minimizes our value
                    new_childval = np.min(grand_vals)
                    qvals[action_to_child] = np.mean([qvals[action_to_child], new_childval])
                else:
                    qvals[action_to_child] = -1

        if np.count_nonzero(qvals) == 0:
            qvals += valid_indicators

        assert step is not None
        if step < self.temp_steps:
            pi = temperate_pi(qvals, self.temp, valid_indicators)
        else:
            pi = temperate_pi(qvals, 0, valid_indicators)
        return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.num_searches}S {self.temp:.4f}T]-{self.name}"


class ActorCritic(Policy):
    def __init__(self, name, network):
        super(ActorCritic, self).__init__(name, temp=0)
        self.pytorch_model = network

    def __call__(self, go_env, step=None):
        """
        :param go_env:
        :param step: Parameter used for getting the temperature
        :return: Action probabilities
        """
        self.pytorch_model.eval()
        state = go_env.get_canonical_state()
        state_tensor = torch.from_numpy(state[np.newaxis]).type(torch.FloatTensor)
        policy_scores, _ = self.pytorch_model(state_tensor)
        pi = torch.nn.functional.softmax(policy_scores, dim=1)
        pi = pi.detach().numpy()[0]
        return pi


RAND_PI = Random()
GREEDY_PI = MCTS('Greedy', greedy_val_func, num_searches=0)
SMART_GREEDY_PI = MCTS('Smart Greedy', smart_greedy_val_func, num_searches=0)
HUMAN_PI = Human()
