import gym
import numpy as np
from sklearn import preprocessing
from scipy import special

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def invert_val(val):
    return -val


def qs_from_valfunc(state, val_func, group_map=None):
    canonical_children, _ = GoGame.get_canonical_children(state, group_map)
    child_vals = val_func(np.array(canonical_children))
    qvals = vals_to_qs(child_vals, state)
    return qvals, canonical_children


def vals_to_qs(canonical_childvals, state):
    valid_moves = GoGame.get_valid_moves(state)
    action_size = GoGame.get_action_size(state)
    qvals = np.zeros(action_size)
    for child_idx, action in enumerate(np.argwhere(valid_moves)):
        child_val = canonical_childvals[child_idx]
        qvals[action] = invert_val(child_val)

    return qvals


def batchqs_from_valfunc(states, val_func, group_maps=None):
    """
    :param states:
    :param val_func:
    :return: qvals of children of every state (batch size x children state vals)
    """
    if group_maps is None:
        group_maps = [None for _ in range(len(states))]
    batch_qvals = []
    batch_canon_children = []
    for state, group_map in zip(states, group_maps):
        qvals, canonical_children = qs_from_valfunc(state, val_func, group_map)
        batch_qvals.append(qvals)
        batch_canon_children.append(canonical_children)

    return np.array(batch_qvals), batch_canon_children


def greedy_pi(qvals, valid_moves):
    expq = np.exp(qvals - np.max(qvals))
    expq *= valid_moves
    max_qs = np.max(expq)
    pi = (expq == max_qs).astype(np.int)
    pi = preprocessing.normalize(pi[np.newaxis], norm='l1')[0]
    return pi


def temperate_pi(qvals, temp, valid_moves):
    if temp <= 0:
        # Max Qs
        pi = greedy_pi(qvals, valid_moves)
    else:
        pi = np.zeros(qvals.shape)
        valid_indcs = np.argwhere(valid_moves)
        pi[valid_indcs] = special.softmax(qvals[valid_indcs] * (1 / temp))

    return pi
