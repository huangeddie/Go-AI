import gym
import numpy as np
import torch
from scipy import special

from go_ai import montecarlo
from go_ai.models import pytorch_val_to_numpy, pytorch_ac_to_numpy
from go_ai.montecarlo import tree

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

    def __init__(self, name, temp=None):
        self.name = name
        self.temp = temp
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


class Value(Policy):
    def __init__(self, name, val_func, mcts, temp=0):
        super(Value, self).__init__(name, temp)
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
        avg_qs = np.nanmean([prior_qs, post_qs], axis=0)
        pi = montecarlo.temp_softmax(avg_qs, self.temp, valid_moves)

        if 'debug' in kwargs:
            if kwargs['debug']:
                return pi, prior_qs, post_qs
        else:
            return pi

    def mcts_qvals(self, go_env):
        canonical_children, child_groupmaps = go_env.get_children(canonical=True)
        child_vals = self.val_func(np.array(canonical_children))
        valid_moves = go_env.get_valid_moves()

        prior_qs = montecarlo.vals_to_qs(child_vals, valid_moves)
        post_qs = np.full(prior_qs.shape, np.nan)

        # Search on grandchildren layer
        if self.mcts > 0:
            argvalid_moves = np.argwhere(valid_moves).flatten()
            assert len(argvalid_moves) == len(child_vals)
            ordered_child_idcs = np.argsort(child_vals.flatten())
            best_child_idcs = ordered_child_idcs[:self.mcts]

            all_grandchildren = []
            all_actions = []
            all_valid_moves = []
            seperators = []
            for child_idx in best_child_idcs:
                child = canonical_children[child_idx]
                if GoGame.get_game_ended(child):
                    continue

                action_to_child = argvalid_moves[child_idx]
                child_groupmap = child_groupmaps[child_idx]
                grandchildren, _ = GoGame.get_children(child, child_groupmap, canonical=True)

                start = len(all_grandchildren)
                all_grandchildren.extend(grandchildren)
                end = len(all_grandchildren)
                all_actions.append(action_to_child)
                seperators.append((start, end))
                all_valid_moves.append(GoGame.get_valid_moves(child))

            if len(all_grandchildren) > 0:
                all_grand_vals = self.val_func(np.array(all_grandchildren))
                for action, sep, valid_moves in zip(all_actions, seperators, all_valid_moves):
                    grand_vals = all_grand_vals[sep[0]: sep[1]].flatten()
                    qs = np.zeros(valid_moves.shape)
                    where_valid = np.where(valid_moves)
                    qs[where_valid] = montecarlo.invert_vals(grand_vals)
                    pi = montecarlo.temp_softmax(qs, self.temp, valid_moves)
                    post_qs[action] = montecarlo.invert_vals(np.inner(pi, qs))
                    post_qs[action] = np.min(grand_vals)

        return prior_qs, post_qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"


class ActorCritic(Policy):
    def __init__(self, name, model, mcts, temp,):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(ActorCritic, self).__init__(name, temp=temp)
        self.pytorch_model = model
        self.ac_func = pytorch_ac_to_numpy(model)
        self.mcts = mcts

    def get_tree(self, go_env):
        _, _, root = tree.mct_search(go_env, self.mcts, self.ac_func)
        return root

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        if self.mcts > 0:
            visits, prior_qs, root = tree.mct_search(go_env, self.mcts, self.ac_func)

            pi = visits ** (1 / self.temp)
            pi = pi / np.sum(pi)

            if 'debug' in kwargs:
                if kwargs['debug']:
                    return pi, prior_qs, visits

        else:
            state = go_env.get_canonical_state()
            policy_scores, _ = self.ac_func(state[np.newaxis])
            policy_scores = policy_scores[0]
            valid_moves = GoGame.get_valid_moves(state)
            pi = montecarlo.temp_softmax(policy_scores, self.temp, valid_moves)

        return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"


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


RAND_PI = Random()
GREEDY_PI = Value('Greedy', greedy_val_func, mcts=0)
SMART_GREEDY_PI = Value('Smart Greedy', smart_greedy_val_func, mcts=0)
HUMAN_PI = Human('terminal')
