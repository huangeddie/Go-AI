import gym
import numpy as np
from sklearn.preprocessing import normalize

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def invert_val(val):
    return -val


def canonical_winning(canonical_state):
    my_area, opp_area = GoGame.get_areas(canonical_state)
    if my_area > opp_area:
        winning = 1
    elif my_area < opp_area:
        winning = -1
    else:
        winning = 0

    return winning


def batch_canonical_children_states(states):
    # Get all children states
    canonical_next_states = []
    for state in states:
        children = GoGame.get_children(state)
        for child in children:
            canonical_child = GoGame.get_canonical_form(child)
            canonical_next_states.append(canonical_child)

    # Get network responses on children
    canonical_next_states = np.array(canonical_next_states)
    return canonical_next_states


def qs_from_stateval(states, val_func):
    """
    :param states:
    :param val_func:
    :return: qvals of children of every state (batch size x children state vals)
    """

    canonical_next_states = batch_canonical_children_states(states)
    canonical_next_vals = val_func(canonical_next_states)

    curr_idx = 0
    batch_qvals = []
    for state in states:
        valid_moves = GoGame.get_valid_moves(state)
        Qs = []
        for move in range(GoGame.get_action_size(state)):
            if valid_moves[move]:
                canonical_next_state = canonical_next_states[curr_idx]
                terminal = GoGame.get_game_ended(canonical_next_state)
                winning = canonical_winning(canonical_next_state)
                oppo_val = (1 - terminal) * canonical_next_vals[curr_idx].item() + (terminal) * winning
                qval = invert_val(oppo_val)
                Qs.append(qval)
                curr_idx += 1
            else:
                Qs.append(0)

        batch_qvals.append(Qs)

    assert curr_idx == len(canonical_next_vals), (curr_idx, len(canonical_next_vals))
    return np.array(batch_qvals), canonical_next_states


def greedy_pi(qvals, valid_moves):
    expq = np.exp(qvals - np.max(qvals))
    expq *= valid_moves
    max_qs = np.max(expq)
    pi = (expq == max_qs).astype(np.int)
    pi = normalize(pi[np.newaxis], norm='l1')[0]
    return pi


def temperate_pi(qvals, temp, valid_moves):
    if temp <= 0:
        # Max Qs
        pi = greedy_pi(qvals, valid_moves)
    else:
        expq = np.exp(qvals - np.max(qvals))
        expq *= valid_moves
        amp_qs = expq[np.newaxis] ** (1 / temp)
        pi = normalize(amp_qs, norm='l1')[0]

    return pi
