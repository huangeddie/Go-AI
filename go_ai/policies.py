from go_ai import mcts, data
from go_ai.models import actor_critic, value_model
import gym
import numpy as np
from sklearn.preprocessing import normalize

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
            print(gogame.str(state))
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
            player_action = gogame.action_2d_to_1d(player_action, state)
            if valid_moves[player_action]:
                break
            else:
                print("Invalid action")

        action_probs = np.zeros(gogame.get_action_size(state))
        action_probs[player_action] = 1
        return action_probs


class GreedyPolicy(Policy):
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def __call__(self, state, step):
        """
        :param state:
        :param step:
        :return:
        """
        valid_moves = gogame.get_valid_moves(state)
        invalid_moves = 1 - valid_moves

        batch_qvals = mcts.qval_from_stateval(state[np.newaxis], self.forward_func)
        qvals = batch_qvals[0]
        qvals -= np.min(qvals)
        qvals += 1e-7
        qvals = qvals * valid_moves

        if step < 4:
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
    def __init__(self, network, state, mc_sims, temp_func=lambda step: (1 / 8) if (step < 16) else 0):
        forward_func = actor_critic.make_forward_func(network)

        self.forward_func = forward_func
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

class ActorCriticPolicy(Policy):
    def __init__(self, network):
        forward_func = actor_critic.make_forward_func(network)

        self.forward_func = forward_func

    def __call__(self, state, step):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        action_probs, _ = self.forward_func(state[np.newaxis])
        return action_probs[0]


def make_policy(policy_args):
    board_size = policy_args['board_size']

    if policy_args['mode'] == 'values':
        model = value_model.make_val_net(board_size)
        model.load_weights(policy_args['model_path'])
        forward_func = value_model.make_forward_func(model)
        policy = GreedyPolicy(forward_func)

    elif policy_args['mode'] == 'actor_critic':
        model = actor_critic.make_actor_critic(board_size)
        model.load_weights(policy_args['model_path'])
        policy = ActorCriticPolicy(actor_critic)

    elif policy_args['mode'] == 'random':
        policy = RandomPolicy()

    elif policy_args['mode'] == 'greedy':
        policy = GreedyPolicy(value_model.greedy_vals)

    else:
        raise Exception("Unknown policy mode")

    return policy
