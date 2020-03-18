import gym
import numpy as np
import torch

from go_ai.models import val_net, ac_net, attn_net
from go_ai.policies import Policy
from go_ai.policies.actorcritic import ActorCritic
from go_ai.policies.value import Value
from go_ai.policies.attn import Attn

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
GREEDY_PI = Value('Greedy', greedy_val_func)
SMART_GREEDY_PI = Value('Smart Greedy', smart_greedy_val_func)
HUMAN_PI = Human('terminal')


def create_policy(args, name=''):
    model = args.model
    size = args.size
    if model == 'val':
        net = val_net.ValueNet(size, args.resblocks)
        pi = Value(name, net, args)
    elif model == 'ac':
        net = ac_net.ActorCriticNet(size, args.resblocks)
        pi = ActorCritic(name, net, args)
    elif model == 'attn':
        net = attn_net.AttnNet(size, args.resblocks)
        pi = Attn(name, net, args)
    elif model == 'rand':
        net = None
        pi = RAND_PI
        return pi, net
    elif model == 'greedy':
        pi = Value('Greedy', greedy_val_func, args)
        return pi, greedy_val_func
    elif model == 'human':
        net = None
        pi = Human(args.render)
        return pi, net
    else:
        raise Exception("Unknown model argument", model)

    load_weights(args, net)

    return pi, net


def load_weights(args, net):
    if args.baseline:
        assert not args.latest_checkpoint
        assert args.customdir == ''
        net.load_state_dict(torch.load(args.basepath, args.device))
    elif args.latest_checkpoint:
        assert not args.baseline
        assert args.customdir == ''
        net.load_state_dict(torch.load(args.checkpath, args.device))
    elif args.customdir != '':
        assert not args.latest_checkpoint
        assert not args.baseline
        net.load_state_dict(torch.load(args.custompath, args.device))
