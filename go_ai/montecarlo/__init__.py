import gym
import numpy as np
from sklearn.preprocessing import normalize

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def invert_qval(qval):
    return 1 - qval


def canonical_winning(canonical_state):
    my_area, opp_area = GoGame.get_areas(canonical_state)
    if my_area > opp_area:
        winning = 1
    elif my_area < opp_area:
        winning = 0
    else:
        winning = 0.5

    return winning


def batch_canonical_children_states(states):
    # Get all children states
    canonical_next_states = []
    for state in states:
        valid_moves = GoGame.get_valid_moves(state)
        valid_move_idcs = np.argwhere(valid_moves > 0).flatten()
        for move in valid_move_idcs:
            next_state = GoGame.get_next_state(state, move)
            canonical_next_state = GoGame.get_canonical_form(next_state)
            canonical_next_states.append(canonical_next_state)
    # Get network responses on children
    canonical_next_states = np.array(canonical_next_states)
    return canonical_next_states


def qval_from_stateval(states, val_func):
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
                qval = invert_qval(oppo_val)
                Qs.append(qval)
                curr_idx += 1
            else:
                Qs.append(0)

        batch_qvals.append(Qs)

    assert curr_idx == len(canonical_next_vals), (curr_idx, len(canonical_next_vals))
    return np.array(batch_qvals), canonical_next_states


def greedy_pi(qvals, valid_moves):
    expq = np.exp(qvals)
    expq *= valid_moves
    max_qs = np.max(expq)
    pi = (expq == max_qs).astype(np.int)
    pi = normalize(pi[np.newaxis], norm='l1')[0]
    return pi


def exp_temp(qvals, temp, valid_moves):
    if temp <= 0:
        # Max Qs
        pi = greedy_pi(qvals, valid_moves)
    else:
        expq = np.exp(qvals)
        expq *= valid_moves
        amp_qs = expq[np.newaxis] ** (1 / temp)
        if np.isnan(amp_qs).any():
            pi = greedy_pi(qvals, valid_moves)
        else:
            pi = normalize(amp_qs, norm='l1')[0]
            if np.count_nonzero(pi) == 0:
                # Incase we amplify so much, everything is zero due to floating point error
                # Max Qs
                pi = greedy_pi(qvals, valid_moves)

    assert (pi[valid_moves == 0] == 0).all(), (qvals, pi, valid_moves)
    return pi
