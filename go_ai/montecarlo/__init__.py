import gym
import numpy as np
from scipy import special
from sklearn import preprocessing

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def invert_vals(vals):
    return -vals


def qs_from_valfunc(state, val_func, group_map=None):
    canonical_children, _ = GoGame.get_children(state, group_map, canonical=True)
    child_vals = val_func(np.array(canonical_children))
    valid_moves = GoGame.get_valid_moves(state)
    qvals = vals_to_qs(child_vals, valid_moves)
    return qvals, canonical_children


def vals_to_qs(canonical_childvals, valid_moves):
    qvals = np.zeros(valid_moves.shape)
    qvals[np.where(valid_moves)] = invert_vals(canonical_childvals.flatten())

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


def batch_greedy_pi(batch_qvals, batch_valid_moves):
    expq = np.exp(batch_qvals - np.max(batch_qvals, axis=1, keepdims=True))
    expq *= batch_valid_moves
    max_qs = np.max(expq, axis=1, keepdims=True)
    pi = (expq == max_qs).astype(np.int)
    pi = preprocessing.normalize(pi, norm='l1')
    return pi


def temp_softmax(qvals, temp, valid_moves):
    if temp <= 0:
        # Max Qs
        pi = greedy_pi(qvals, valid_moves)
    else:
        pi = np.zeros(valid_moves.shape)
        valid_indcs = np.where(valid_moves)
        if qvals.shape == valid_moves.shape:
            qvals = qvals[valid_indcs]
        pi[valid_indcs] = special.softmax(qvals * (1 / temp))

    return pi


def temperature(qs, temp, valid_moves):
    if temp <= 0:
        pi = greedy_pi(qs, valid_moves)
    else:
        pi = np.zeros(valid_moves.shape)
        where_valid = np.where(valid_moves)
        pi[where_valid] = preprocessing.normalize(qs[where_valid][np.newaxis], norm='l1')[0]
        pi = preprocessing.normalize(pi[np.newaxis] ** (1 / temp), norm='l1')[0]

    return pi


def batch_temperate_pi(batch_qvals, temp, batch_valid_moves):
    if temp <= 0:
        # Max Qs
        pi = batch_greedy_pi(batch_qvals, batch_valid_moves)
    else:
        pi = special.softmax(batch_qvals * (1 / temp), axis=1)

    return pi
