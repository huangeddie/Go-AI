from go_ai import models, mcts
import gym
import numpy as np

go_env = gym.make('gym_go:go-v0', size=0)
gogame = go_env.gogame

class Policy:
    """
    Interface for all types of policies
    """
    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        pass

    def step(self, action):
        """
        Helps synchronize the policy with the outside environment
        :param action:
        :return:
        """
        pass

    def reset(self):
        """
        Helps synchronize the policy with the outside environment
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
        valid_moves = gogame.get_valid_moves(state)
        return valid_moves / np.sum(valid_moves)

class HumanPolicy(Policy):
    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return: Action probabilities
        """
        valid_moves = gogame.get_valid_moves(state)
        while True:
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
            player_action = go_env.action_2d_to_1d(player_action)
            if valid_moves[player_action]:
                break
            else:
                print("Invalid action")

        action_probs = np.zeros(gogame.get_action_size(state))
        action_probs[player_action] = 1
        return action_probs

class MctPolicy(Policy):
    def __init__(self, network, state, mc_sims, temp_func):
        self.forward_func = lambda states: models.forward_pass(states, network, training=False)
        self.mc_sims = mc_sims
        self.temp_func = temp_func
        self.tree = mcts.MCTree(state, self.forward_func)

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