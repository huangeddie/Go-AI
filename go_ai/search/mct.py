import gym
import numpy as np
from scipy import special

from go_ai.search import tree

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def val_search(go_env, mcts, val_func):
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
            curr_node.make_children()
            next_nodes = curr_node.get_child_nodes()
            tree.set_state_vals(val_func, next_nodes)
            curr_node.set_val_prior()

            move = np.argmax(curr_node.prior_pi)
            curr_node = curr_node.step(move)

        curr_node.backprop(curr_node.val)

    return rootnode

def backprop_winning(node):
    winning = node.winning()
    if node.level % 2 == 1:
        val = -winning
    else:
        val = winning
    node.backprop(val)

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
    children = rootnode.make_children()

    root_prior_logits, root_val_logits = ac_func(rootstate[np.newaxis], children[np.newaxis])
    root_prior_pi = special.softmax(root_prior_logits.flatten())

    rootnode.backprop(np.tanh(root_val_logits).item())
    rootnode.set_prior_pi(root_prior_pi)

    found_winnode = None
    child_nodes = rootnode.get_child_nodes()
    for node in child_nodes:
        if node.terminal() and node.winning() == -1:
            found_winnode = node
            break

    if found_winnode is not None:
        for child in child_nodes:
            child.visits = 0
        found_winnode.visits = 1
        return rootnode

    # MCT Search
    v = 4
    for i in range(0, num_searches, v):
        curr_node = rootnode
        while curr_node.visits > 0 and not curr_node.terminal():
            ucbs = curr_node.get_ucbs()
            move = np.argmax(ucbs)
            curr_node = curr_node.step(move)

        if curr_node.terminal():
            backprop_winning(curr_node)
        else:
            parent = curr_node.parent
            nvalid = int(sum(parent.valid_moves()))
            ucbs = np.array(parent.get_ucbs())
            best_moves = np.argsort(-ucbs)[:min(v, nvalid)]
            best_nodes = []
            for move in best_moves:
                node = parent.step(move)
                if node.terminal():
                    backprop_winning(node)
                else:
                    best_nodes.append(node)
            best_states = list(map(lambda node: node.state, best_nodes))
            children = list(map(lambda node: node.make_children(), best_nodes))
            all_prior_qs, all_logits = ac_func(np.array(best_states), children)

            for node, prior_qs, logit, in zip(best_nodes, all_prior_qs, all_logits):
                prior_pi = special.softmax(prior_qs)
                logit = logit.item()
                val = np.tanh(logit)
                node.set_prior_pi(prior_pi)
                node.backprop(val)

    return rootnode
