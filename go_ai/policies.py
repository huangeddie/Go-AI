import logging

import gym
import numpy as np
import torch

from go_ai import montecarlo
from go_ai.montecarlo import tree, exp_temp

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
                val = 0
            else:
                val = 0
        else:
            val = (black_area - white_area + board_area) / (2 * board_area)
        vals.append(val)
    vals = np.array(vals, dtype=np.float)
    return vals[:, np.newaxis]


def pytorch_to_numpy(model, logits):
    """
    Note: For now everything is assumed to be on CPU
    :param model:
    :return: The numpy equivalent of the pytorch value model
    """

    def val_func(states):
        model.eval()
        with torch.no_grad():
            states = torch.from_numpy(states).type(torch.FloatTensor)
            state_vals = model(states)
            if logits:
                state_vals = torch.sigmoid(state_vals)
            return state_vals.numpy()

    return val_func


class Policy:
    """
    Interface for all types of policies
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, state, step):
        """
        :param state: Go board
        :param step: the number of steps taken in the game so far
        :return: Action probabilities
        """
        pass

    def step(self, action):
        """
        Helps synchronize the policy with the outside environment.
        Most policies don't need to implement this function
        :param action:
        :return:
        """
        pass

    def reset(self, state=None):
        """
        Helps synchronize the policy with the outside environment.
        Most policies don't need to implement this function
        :return:
        """
        pass


class RandomPolicy(Policy):
    def __init__(self):
        super(RandomPolicy, self).__init__('Random')

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        valid_moves = GoGame.get_valid_moves(state)
        return valid_moves / np.sum(valid_moves)


class HumanPolicy(Policy):
    def __init__(self):
        super(HumanPolicy, self).__init__('Human')

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        valid_moves = GoGame.get_valid_moves(state)

        # Human interface
        size = state.shape[1]
        go_env = gym.make('gym_go:go-v0', size=size, state=state)
        while True:
            player_action = go_env.render('human')
            player_action = GoGame.action_2d_to_1d(player_action, state)
            if valid_moves[player_action] > 0:
                break

        action_probs = np.zeros(GoGame.get_action_size(state))
        action_probs[player_action] = 1

        return action_probs


class QTempPolicy(Policy):
    def __init__(self, name, val_func, temp):
        """
        Pi is proportional to the exp(qvals) raised to the 1/temp power
        :param val_func: A function that takes in states and outputs corresponding values
        """

        super(QTempPolicy, self).__init__(name)

        if isinstance(val_func, torch.nn.Module):
            logging.info("Converting pytorch value function to numpy")
            val_func = pytorch_to_numpy(val_func, logits=True)

        self.val_func = val_func
        self.temp = temp

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return:
        """
        valid_moves = GoGame.get_valid_moves(state)
        invalid_moves = 1 - valid_moves

        batch_qvals, _ = montecarlo.qval_from_stateval(state[np.newaxis], self.val_func)
        qvals = batch_qvals[0]
        if np.count_nonzero(qvals) == 0:
            qvals += valid_moves
        temp = self.temp

        pi = exp_temp(qvals, temp, valid_moves)

        assert (pi[invalid_moves > 0] == 0).all(), pi
        return pi


class MctPolicy(Policy):
    def __init__(self, name, board_size, val_func, temp, num_searches):
        super(MctPolicy, self).__init__(name)
        if isinstance(val_func, torch.nn.Module):
            logging.info("Converting pytorch value function to numpy")
            val_func = pytorch_to_numpy(val_func, logits=True)

        self.val_func = val_func
        self.temp = temp
        initial_state = GoGame.get_init_board(board_size)
        self.tree = tree.MCTree(self.val_func, initial_state)
        self.num_searches = num_searches

    def __call__(self, state, step):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        root = self.tree.root.state
        assert (root == state).all(), (root, state)
        pi = self.tree.get_action_probs(max_num_searches=self.num_searches, temp=self.temp)
        root = self.tree.root.state
        assert (root == state).all(), (root, state)
        return pi

    def step(self, action):
        """
        Helps synchronize the policy with the outside environment
        :param action:
        :return:
        """
        self.tree.step(action)

    def reset(self, state=None):
        self.tree.reset(state)
