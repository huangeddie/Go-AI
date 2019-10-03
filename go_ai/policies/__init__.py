import go_ai.montecarlo
from go_ai.montecarlo import tree
import gym
import numpy as np
from sklearn.preprocessing import normalize

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class Policy:
    """
    Interface for all types of policies
    """

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
    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        valid_moves = GoGame.get_valid_moves(state)
        return valid_moves / np.sum(valid_moves)


class HumanPolicy(Policy):
    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        valid_moves = GoGame.get_valid_moves(state)

        # Human interface
        size = state.shape[1]
        action_size = GoGame.get_action_size(state)
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
    def __init__(self, val_func, temp):
        """
        Pi is proportional to the qvals raised to the 1/temp power
        :param val_func: A function that takes in states and outputs corresponding values
        """
        self.val_func = val_func
        self.temp = temp

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return:
        """
        max_steps = 2 * GoGame.get_action_size(state)
        valid_moves = GoGame.get_valid_moves(state)
        invalid_moves = 1 - valid_moves

        batch_qvals = go_ai.montecarlo.qval_from_stateval(state[np.newaxis], self.val_func)
        qvals = batch_qvals[0]
        if np.count_nonzero(qvals) == 0:
            qvals += valid_moves

        augment = int(((step + 1) / max_steps) * (1 / self.temp)) if self.temp > 0 else np.PINF
        aug_qs = qvals[np.newaxis] ** augment

        if self.temp <= 0 or np.count_nonzero(aug_qs) == 0:
            # Max Q
            max_qs = np.max(qvals)
            pi = (qvals == max_qs).astype(np.int)
            pi = normalize(pi[np.newaxis], norm='l1')[0]
        else:
            pi = normalize(aug_qs, norm='l1')[0]

        assert (pi[invalid_moves > 0] == 0).all(), pi
        return pi


class MctPolicy(Policy):
    def __init__(self, forward_func, state, temp_func=lambda step: 1 if (step < 4) else 0):
        self.forward_func = forward_func
        self.temp_func = temp_func
        self.tree = tree.MCTree(self.forward_func, state)

    def __call__(self, state, step):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        temp = self.temp_func(step)
        return self.tree.get_action_probs(max_num_searches=0, temp=temp)

    def step(self, action):
        """
        Helps synchronize the policy with the outside environment
        :param action:
        :return:
        """
        self.tree.step(action)

    def reset(self, state=None):
        self.tree.reset(state)


class PolicyArgs:
    def __init__(self, mode, board_size, weight_path=None, name=None, temperature=None):
        self.mode = mode
        self.board_size = board_size
        self.model_path = weight_path
        self.name = name if name is not None else mode
        self.temperature = temperature
