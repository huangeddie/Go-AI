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
                curr_node.set_prior_pi(None)

            ucbs = curr_node.get_ucbs()
            move = np.argmax(ucbs)
            curr_node = curr_node.step(move)

        if not curr_node.terminal():
            curr_node.make_children()
            next_nodes = curr_node.get_child_nodes()
            tree.set_state_vals(val_func, next_nodes)
            curr_node.set_prior_pi(None)

            move = np.argmax(curr_node.prior_pi)
            curr_node = curr_node.step(move)

        curr_node.backprop(curr_node.val)

    return rootnode


def backprop_winning(node):
    winning = node.winning()
    node.backprop(winning)


def find_next_node(node):
    curr = node
    while curr.visits > 0 and not curr.terminal():
        ucbs = curr.get_ucbs()
        move = np.nanargmax(ucbs)
        curr = curr.step(move)

    return curr

def mct_search(go_env, num_searches, actor_critic=None, critic=None):
    # Setup the root
    root_group_map = go_env.get_canonical_group_map()
    rootstate = go_env.get_canonical_state()
    rootnode = tree.Node(rootstate, root_group_map)

    # The first iteration doesn't count towards the number of searches
    if actor_critic is not None:
        root_prior_logits, root_val_logits = actor_critic(rootstate[np.newaxis])
        root_prior_pi = special.softmax(root_prior_logits.flatten())
        rootnode.set_prior_pi(root_prior_pi)
    else:
        assert critic is not None
        root_val_logits = critic(rootstate[np.newaxis])

    rootnode.backprop(np.tanh(root_val_logits).item())

    # MCT Search
    for i in range(0, num_searches):
        # Get top k nodes to visit
        node = find_next_node(rootnode)

        # Terminal case
        if node.terminal():
            backprop_winning(node)
            continue

        # Compute pi's and values on internal nodes
        pi_logits, val_logits = actor_critic(node.state[np.newaxis])

        # Prior Pi
        pi = special.softmax(pi_logits.flatten())
        node.set_prior_pi(pi)

        # Backprop value
        val = np.tanh(val_logits.item())
        node.backprop(val)

    assert rootnode.visits > 1
    return rootnode
