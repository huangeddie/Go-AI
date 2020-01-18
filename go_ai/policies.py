from queue import Queue

import gym
import numpy as np
import torch

from go_ai import search
from go_ai.models import pytorch_val_to_numpy, pytorch_ac_to_numpy
from go_ai.search import mct

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
                val = 100
            elif black_area < white_area:
                val = -100
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
        self.depth = int(np.log2(self.mcts)) if self.mcts > 0 else 0

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        rootnode = mct.val_search(go_env, self.mcts, self.val_func)
        qs = self.tree_to_qs(rootnode)

        exp_qs = np.exp(qs)
        pi = np.nansum(exp_qs, axis=0)
        pi = search.temp_norm(pi, self.temp, rootnode.valid_moves)

        if 'debug' in kwargs:
            if kwargs['debug']:
                return pi, rootnode
        else:
            return pi

    def tree_to_qs(self, rootnode):
        qs = np.full((self.depth + 1, rootnode.actionsize()), np.nan)

        # Iterate through root-child nodes, inner nodes
        queue = Queue()
        for child in rootnode.get_real_children():
            queue.put(child)

        while not queue.empty():
            node = queue.get()
            if node.level == 1:
                # Root child
                qs[0, node.first_action] = search.invert_vals(node.val)

            if not node.isleaf():
                # Inner node
                likely_node = min(node.get_real_children(), key=lambda node: node.val)
                level = likely_node.level

                orig_qval = qs[level - 1, node.first_action]
                if level % 2 == 0:
                    qval = np.nanmin([likely_node.val, orig_qval])
                else:
                    qval = np.nanmax([search.invert_vals(likely_node.val), orig_qval])

                qs[level - 1, node.first_action] = qval

                # Queue children
                for child in node.get_real_children():
                    queue.put(child)
        return qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"


class ActorCritic(Policy):
    def __init__(self, name, model, mcts, temp, ):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(ActorCritic, self).__init__(name, temp=temp)
        self.pytorch_model = model
        self.ac_func = pytorch_ac_to_numpy(model)
        self.mcts = mcts

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        if self.mcts > 0:
            rootnode = mct.ac_search(go_env, self.mcts, self.ac_func)
            qs = self.tree_to_qs(rootnode)

            pi = qs[1] ** (1 / self.temp)
            pi = pi / np.sum(pi)

            if 'debug' in kwargs:
                if kwargs['debug']:
                    return pi, rootnode

        else:
            state = go_env.get_canonical_state()
            policy_scores, _ = self.ac_func(state[np.newaxis])
            policy_scores = policy_scores[0]
            valid_moves = GoGame.get_valid_moves(state)
            pi = search.temp_softmax(policy_scores, self.temp, valid_moves)

        return pi

    def tree_to_qs(self, rootnode):
        qs = np.empty((2, rootnode.actionsize))
        qs[0] = rootnode.prior_pi
        qs[1] = rootnode.get_visit_counts()

        return qs

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
