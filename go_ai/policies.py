import gym
import numpy as np
import torch

from go_ai import montecarlo
from go_ai.models import pytorch_val_to_numpy, pytorch_ac_to_numpy
from go_ai.montecarlo import tree
from scipy import special

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
            area_val = (black_area - white_area) / (board_area)
            libs_val = (blacklibs - whitelibs) / (board_area)
            val = (6 * area_val + libs_val) / 7
        vals.append(val)
    vals = np.array(vals, dtype=np.float)
    return vals[:, np.newaxis]


class Policy:
    """
    Interface for all types of policies
    """

    def __init__(self, name, temp=None, temp_steps=None):
        self.name = name
        self.temp = temp
        self.temp_steps = temp_steps
        self.pytorch_model = None

    def __call__(self, go_env, **kwargs):
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

    def __call__(self, go_env, **kwargs):
        """
        :param go_env:
        :param step:
        :return: Action probabilities
        """

        valid_moves = go_env.get_valid_moves()
        return valid_moves / np.sum(valid_moves)


class Human(Policy):
    def __init__(self, render):
        self.render = render
        super(Human, self).__init__('Human')

    def __call__(self, go_env, **kwargs):
        """
        :param go_env:
        :param step:
        :return: Action probabilities
        """
        state = go_env.get_state()
        valid_moves = go_env.get_valid_moves()

        # Human interface
        if self.render == 'human':
            while True:
                player_action = go_env.render(self.render)
                if player_action is None:
                    player_action = go_env.action_space.n - 1
                else:
                    player_action = GoGame.action_2d_to_1d(player_action, state)
                if valid_moves[player_action] > 0:
                    break
        else:
            while True:
                try:
                    go_env.render(self.render)
                    coor = input("Enter actions coordinates i j:\n")
                    if coor == 'p':
                        player_action = None
                    elif coor == 'e':
                        player_action = None
                        exit()
                    else:
                        coor = coor.split(' ')
                        player_action = (int(coor[0]), int(coor[1]))

                    player_action = GoGame.action_2d_to_1d(player_action, state)
                    if valid_moves[player_action] > 0:
                        break
                except Exception:
                    pass

        action_probs = np.zeros(GoGame.get_action_size(state))
        action_probs[player_action] = 1

        return action_probs


class Value(Policy):
    def __init__(self, name, val_func, mcts, temp=0, tempsteps=0):
        super(Value, self).__init__(name, temp, tempsteps)
        if isinstance(val_func, torch.nn.Module):
            self.pytorch_model = val_func
            val_func = pytorch_val_to_numpy(val_func)

        self.val_func = val_func
        self.mcts = mcts

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """

        prior_qs, post_qs = self.mcts_qvals(go_env)
        valid_moves = go_env.get_valid_moves()
        where_valid = np.where(valid_moves)
        prior_pi, post_pi = np.zeros(prior_qs.shape), np.zeros(post_qs.shape)
        prior_pi[where_valid] = special.softmax(prior_qs[where_valid])
        post_pi[where_valid] = special.softmax(post_qs[where_valid])
        pi = np.nansum([prior_pi, post_pi], axis=0)

        step = kwargs['step']
        assert step is not None
        if step < self.temp_steps:
            pi = pi ** (1 / self.temp)
        else:
            pi = pi ** (1 / 0.01)

        pi /= np.sum(pi)

        if 'get_qs' in kwargs:
            if kwargs['get_qs']:
                return pi, prior_qs, post_qs
        else:
            return pi

    def mcts_qvals(self, go_env):
        canonical_children, child_groupmaps = go_env.get_children(canonical=True)
        child_vals = self.val_func(np.array(canonical_children))
        canonical_state = go_env.get_canonical_state()
        valid_indicators = go_env.get_valid_moves()

        prior_qs = montecarlo.vals_to_qs(child_vals, canonical_state)
        post_qs = np.full(prior_qs.shape, np.nan)

        # Search on grandchildren layer
        if self.mcts > 0:
            valid_moves = np.argwhere(valid_indicators).flatten()
            assert len(valid_moves) == len(child_vals)
            ordered_child_idcs = np.argsort(child_vals.flatten())
            best_child_idcs = ordered_child_idcs[:self.mcts]
            for child_idx in best_child_idcs:
                action_to_child = valid_moves[child_idx]
                child = canonical_children[child_idx]
                if GoGame.get_game_ended(child):
                    continue
                child_groupmap = child_groupmaps[child_idx]
                grandchildren, _ = GoGame.get_children(child, child_groupmap, canonical=True)
                grand_vals = self.val_func(np.array(grandchildren))
                # Assume opponent would take action that minimizes our value
                post_qs[action_to_child] = np.min(grand_vals)

        return prior_qs, post_qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"


class ActorCritic(Policy):
    def __init__(self, name, model, mcts, temp, tempsteps):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(ActorCritic, self).__init__(name, temp=temp, temp_steps=tempsteps)
        self.pytorch_model = model
        self.q_func, self.val_func = pytorch_ac_to_numpy(model)
        self.mcts = mcts

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        if self.mcts > 0:
            qs = tree.mct_search(go_env, self.mcts, self.val_func, self.q_func)

            step = kwargs['step']
            assert step is not None
            if step < self.temp_steps:
                pi = qs * (1 / self.temp)

            else:
                max_visit = np.max(qs)
                pi = qs == max_visit

            pi = pi / np.sum(pi)

            if 'get_qs' in kwargs:
                if kwargs['get_qs']:
                    state = go_env.get_canonical_state()
                    prior_qs = self.q_func(state[np.newaxis])[0]
                    return pi, prior_qs, qs

        else:
            state = go_env.get_canonical_state()
            policy_scores = self.q_func(state[np.newaxis])[0]
            valid_moves = GoGame.get_valid_moves(state)
            pi = montecarlo.temp_softmax(policy_scores, self.temp, valid_moves)

        return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"


RAND_PI = Random()
GREEDY_PI = Value('Greedy', greedy_val_func, mcts=0)
SMART_GREEDY_PI = Value('Smart Greedy', smart_greedy_val_func, mcts=0)
HUMAN_PI = Human('terminal')
