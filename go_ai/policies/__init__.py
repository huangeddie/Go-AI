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

    def reset(self):
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
        player_action = None
        while True:
            print(GoGame.str(state))
            coords = input("Enter coordinates separated by space (`q` to quit)\n")
            if coords == 'p':
                player_action = None
            else:
                try:
                    coords = coords.split()
                    row = int(coords[0])
                    col = int(coords[1])
                    player_action = (row, col)
                except Exception as e:
                    print(e)
                    continue
            player_action = GoGame.action_2d_to_1d(player_action, state)
            if valid_moves[player_action]:
                break
            else:
                print("Invalid action")

        action_probs = np.zeros(GoGame.get_action_size(state))
        action_probs[player_action] = 1
        return action_probs


class GreedyPolicy(Policy):
    def __init__(self, val_func):
        """
        :param val_func: A function that takes in states and outputs corresponding values
        """
        self.val_func = val_func

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return:
        """
        valid_moves = GoGame.get_valid_moves(state)
        invalid_moves = 1 - valid_moves

        batch_qvals = go_ai.montecarlo.qval_from_stateval(state[np.newaxis], self.val_func)
        qvals = batch_qvals[0]
        qvals -= np.min(qvals)
        qvals += 1e-7
        qvals = qvals * valid_moves

        if step < 8:
            # Temperated
            target_pis = normalize(qvals[np.newaxis], norm='l1')[0]
        else:
            # Max Q
            max_qs = np.max(qvals)
            target_pis = (qvals == max_qs).astype(np.int)
            target_pis = normalize(target_pis[np.newaxis], norm='l1')[0]

        assert (target_pis[invalid_moves > 0] == 0).all(), target_pis
        return target_pis


class MctPolicy(Policy):
    def __init__(self, forward_func, state, mc_sims, temp_func=lambda step: (1 / 8) if (step < 16) else 0):
        self.forward_func = forward_func
        self.mc_sims = mc_sims
        self.temp_func = temp_func
        self.tree = tree.MCTree(state, self.forward_func)

    def __call__(self, state, step):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        temp = self.temp_func(step)
        return self.tree.get_action_probs(max_num_searches=self.mc_sims, temp=temp)

    def step(self, action):
        """
        Helps synchronize the policy with the outside environment
        :param action:
        :return:
        """
        self.tree.step(action)

    def reset(self):
        self.tree.reset()


class PolicyArgs:
    def __init__(self, mode, board_size, weight_path=None, name=None):
        self.mode = mode
        self.board_size = board_size
        self.model_path = weight_path
        self.name = name if name is not None else mode