import gym
import numpy as np
from scipy import special

from go_ai import montecarlo, data
from go_ai.montecarlo.node import Node

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def mct_search(go_env, num_searches, ac_func):
    '''
    Description:
        Select a child node that maximizes Q + U,
    Args:
        num_searches (int): maximum number of searches performed
        temp (number): temperature constant
    Returns:
        pi (1d np array): the search probabilities
        num_search (int): number of search performed
    '''
    root_groupmap = go_env.group_map
    rootstate = go_env.get_canonical_state()

    rootnode = Node(rootstate, root_groupmap)
    _, root_val_logits = ac_func(rootstate[np.newaxis])
    rootnode.backprop(np.tanh(root_val_logits).item())

    canonical_children, child_groupmaps = GoGame.get_children(rootstate, root_groupmap, canonical=True)
    child_priors, child_logits = ac_func(canonical_children)
    inv_child_logits = montecarlo.invert_vals(child_logits).flatten()
    inv_child_vals = np.tanh(inv_child_logits)
    child_valids = rootnode.valid_moves
    root_prior_pi, root_prior_logits = np.zeros(rootnode.actionsize), np.zeros(rootnode.actionsize)
    root_prior_logits[np.where(child_valids)] = inv_child_logits
    root_prior_pi[np.where(child_valids)] = special.softmax(inv_child_logits)
    rootnode.set_prior_pi(root_prior_pi)

    batch_valid_moves = data.batch_valid_moves(canonical_children)
    batch_invalid_values = data.batch_invalid_values(canonical_children)
    batch_prior_pi = special.softmax(child_priors * batch_valid_moves + batch_invalid_values, axis=1)

    pbar = zip(canonical_children, child_groupmaps, np.argwhere(child_valids), batch_prior_pi, inv_child_vals)
    for child, groupmap, action, prior_pi, inv_childval in pbar:
        child = Node(child, groupmap, rootnode)
        rootnode.canon_children[action] = child

        child.set_prior_pi(prior_pi)
        child.backprop(inv_childval.item())

    # MCT Search
    remaining_searches = max(num_searches - int(np.sum(child_valids)), 0)
    for i in range(remaining_searches):
        curr_node = rootnode
        while curr_node.visits > 0 and not curr_node.terminal:
            ucbs = curr_node.get_ucbs()
            move = np.argmax(ucbs)
            curr_node = curr_node.traverse(move)

        if curr_node.terminal:
            winning = curr_node.winning
            if curr_node.height % 2 == 1:
                val = -winning
            else:
                val = winning
            curr_node.backprop(val)
        else:
            leaf_state = curr_node.state
            prior_qs, logit = ac_func(leaf_state[np.newaxis])
            prior_qs = prior_qs[0]
            logit = logit.item()

            val = np.tanh(logit)
            if curr_node.height % 2 == 1:
                val = montecarlo.invert_vals(val)

            valid_moves = GoGame.get_valid_moves(leaf_state)
            invalid_values = data.batch_invalid_values(leaf_state[np.newaxis])[0]
            prior_pi = special.softmax(prior_qs * valid_moves + invalid_values)

            curr_node.set_prior_pi(prior_pi)
            curr_node.backprop(val)

    return rootnode.get_visit_counts(), root_prior_logits, rootnode


