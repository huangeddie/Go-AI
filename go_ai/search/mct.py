import gym
import numpy as np
from scipy import special

from go_ai import search, data
from go_ai.search import tree

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def val_search(go_env, mcts, val_func, keep_tree=False):
    rootnode = tree.Node(go_env.state, go_env.group_map)

    for _ in range(mcts + 1):
        curr_node = rootnode
        while curr_node.visits > 0 and not curr_node.terminal():
            if curr_node.isleaf():
                next_nodes = curr_node.make_children()
                tree.set_state_vals(val_func, next_nodes)
                curr_node.set_val_prior()

            ucbs = curr_node.get_ucbs()
            move = np.argmax(ucbs)
            curr_node = curr_node.step(move)

        if not curr_node.terminal():
            next_nodes = curr_node.make_children()
            tree.set_state_vals(val_func, next_nodes)
            curr_node.set_val_prior()

            move = np.argmax(curr_node.prior_pi)
            curr_node = curr_node.step(move)

        curr_node.backprop(curr_node.val)

    return rootnode


def ac_search(go_env, num_searches, ac_func):
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
    root_group_map = go_env.group_map
    rootstate = go_env.get_canonical_state()

    rootnode = tree.Node(rootstate, root_group_map)
    _, root_val_logits = ac_func(rootstate[np.newaxis])
    rootnode.backprop(np.tanh(root_val_logits).item())

    canonical_children, child_group_maps = GoGame.get_children(rootstate, root_group_map, canonical=True)
    child_priors, child_logits = ac_func(canonical_children)
    inv_child_logits = search.invert_vals(child_logits).flatten()
    inv_child_vals = np.tanh(inv_child_logits)
    child_valids = rootnode.valid_moves()
    root_prior_pi, root_prior_logits = np.zeros(rootnode.actionsize()), np.zeros(rootnode.actionsize())
    root_prior_logits[np.where(child_valids)] = inv_child_logits
    root_prior_pi[np.where(child_valids)] = special.softmax(inv_child_logits)
    rootnode.set_prior_pi(root_prior_pi)

    batch_valid_moves = data.batch_valid_moves(canonical_children)
    batch_invalid_values = data.batch_invalid_values(canonical_children)
    batch_prior_pi = special.softmax(child_priors * batch_valid_moves + batch_invalid_values, axis=1)

    pbar = zip(canonical_children, child_group_maps, np.argwhere(child_valids), batch_prior_pi, inv_child_vals)
    for child_state, group_map, action, prior_pi, inv_childval in pbar:
        child = rootnode.make_child(action, child_state, group_map)

        child.set_prior_pi(prior_pi)
        child.backprop(inv_childval.item())

    # MCT Search
    remaining_searches = max(num_searches - int(np.sum(child_valids)), 0)
    for i in range(remaining_searches):
        curr_node = rootnode
        while curr_node.visits > 0 and not curr_node.terminal():
            ucbs = curr_node.get_ucbs()
            move = np.argmax(ucbs)
            curr_node = curr_node.step(move)

        if curr_node.terminal():
            winning = curr_node.winning()
            if curr_node.level % 2 == 1:
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
            if curr_node.level % 2 == 1:
                val = search.invert_vals(val)

            valid_moves = GoGame.get_valid_moves(leaf_state)
            invalid_values = data.batch_invalid_values(leaf_state[np.newaxis])[0]
            prior_pi = special.softmax(prior_qs * valid_moves + invalid_values)

            curr_node.set_prior_pi(prior_pi)
            curr_node.backprop(val)

    return rootnode
