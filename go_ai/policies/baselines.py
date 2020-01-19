import os

import gym
import numpy as np
import torch

from go_ai.models import value, actorcritic, get_modelpath
from go_ai.policies import Policy
from go_ai.policies.actorcritic import ActorCritic
from go_ai.policies.value import Value

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
GREEDY_PI = Value('Greedy', greedy_val_func, mcts=0)
SMART_GREEDY_PI = Value('Smart Greedy', smart_greedy_val_func, mcts=0)
HUMAN_PI = Human('terminal')


def create_policy(args, name, baseline=False, latest_checkpoint=False, modeldir=None):
    model = args.model
    size = args.boardsize
    if model == 'val':
        net = value.ValueNet(size, args.resblocks)
        pi = Value(name, net, args.mcts, args.temp)
    elif model == 'ac':
        net = actorcritic.ActorCriticNet(size, args.resblocks)
        pi = ActorCritic(name, net, args.mcts, args.temp)
    elif model == 'rand':
        net = None
        pi = RAND_PI
        return pi, net
    elif model == 'greedy':
        pi = Value('Greedy', greedy_val_func, args.mcts, args.temp)
        return pi, greedy_val_func
    elif model == 'human':
        net = None
        pi = Human(args.render)
        return pi, net
    else:
        raise Exception("Unknown model argument", model)

    if baseline:
        assert not latest_checkpoint
        net.load_state_dict(torch.load(f'bin/baselines/{model}{size}.pt'))
    elif latest_checkpoint:
        assert not baseline
        assert modeldir is None
        latest_checkpath = get_modelpath(args, 'checkpoint')
        net.load_state_dict(torch.load(latest_checkpath))
    elif modeldir is not None:
        assert not latest_checkpoint
        checkpath = os.path.join(modeldir, f'{model}{size}.pt')
        net.load_state_dict(torch.load(checkpath))

    return pi, net