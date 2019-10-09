import gym
import numpy as np

GoGame = gym.make('gym_go:go-v0', size=0).gogame


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
            next_turn = GoGame.get_turn(next_state)
            canonical_next_state = GoGame.get_canonical_form(next_state, next_turn)
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
                val = (1 - terminal) * canonical_next_vals[curr_idx].item() + (terminal) * winning
                Qs.append(1 - val)
                curr_idx += 1
            else:
                Qs.append(0)

        batch_qvals.append(Qs)

    assert curr_idx == len(canonical_next_vals), (curr_idx, len(canonical_next_vals))
    return np.array(batch_qvals)


def piqval_from_actorcritic(states, forward_func):
    """
    :param states:
    :param forward_func:
    :return: policies and qvals of children of every state
    (batch size x children x pi), (batch size x children state vals)
    """

    # Get network responses on children
    canonical_next_states = batch_canonical_children_states(states)
    canonical_pis, canonical_next_vals = forward_func(canonical_next_states)

    curr_idx = 0
    batch_qvals = []
    batch_pis = []
    for state in states:
        valid_moves = GoGame.get_valid_moves(state)
        Qs = []
        children_pis = []
        for move in range(GoGame.get_action_size(state)):
            if valid_moves[move]:
                canonical_next_state = canonical_next_states[curr_idx]
                terminal = GoGame.get_game_ended(canonical_next_state)
                winning = canonical_winning(canonical_next_state)
                val = (1 - terminal) * canonical_next_vals[curr_idx].item() + (terminal) * winning
                Qs.append(1 - val)
                children_pis.append(canonical_pis[curr_idx])
                curr_idx += 1
            else:
                Qs.append(0)

        batch_qvals.append(Qs)
        batch_pis.append(children_pis)

    assert curr_idx == len(canonical_next_vals), (curr_idx, len(canonical_next_vals))
    assert curr_idx == len(canonical_pis), (curr_idx, len(canonical_pis))
    return np.array(batch_pis), np.array(batch_qvals), canonical_next_states
