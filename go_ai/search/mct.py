import gym
import numpy as np
from scipy import special

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
    root_group_map = go_env.get_canonical_group_map()
    rootstate = go_env.get_canonical_state()

    rootnode = tree.Node(rootstate, root_group_map)
    root_prior_logits, root_val_logits = ac_func(rootstate[np.newaxis])
    root_prior_pi = special.softmax(root_prior_logits.flatten())

    rootnode.backprop(np.tanh(root_val_logits).item())
    rootnode.set_prior_pi(root_prior_pi)

    # Cache searches on immediate children
    children = rootnode.make_children()
    states = list(map(lambda node: node.state, children))
    child_prior_logits, child_val_logits = ac_func(np.array(states))
    child_pi = special.softmax(child_prior_logits, axis=1)
    child_vals = np.tanh(child_val_logits)
    found_winnode = None
    for pi, val, node in zip(child_pi, child_vals, children):
        if node.terminal() and node.winning() == -1:
            found_winnode = node
            break
        node.set_prior_pi(pi)
        node.backprop(val.item())

    if found_winnode is not None:
        for child in children:
            child.visits = 0
        found_winnode.visits = 1
        return rootnode

    remaining_searches = num_searches - len(children)

    # MCT Search
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

            prior_pi = special.softmax(prior_qs)

            curr_node.set_prior_pi(prior_pi)
            curr_node.backprop(val)

    return rootnode
