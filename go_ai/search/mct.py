import gym
import numpy as np
from scipy import special

from go_ai.search import tree

GoGame = gym.make('gym_go:go-v0', size=0).gogame


def find_next_node(node):
    curr = node
    while curr.visits > 0 and not curr.terminal():
        ucbs = curr.get_ucbs()
        move = np.nanargmax(ucbs)
        curr = curr.step(move)

    return curr


def mct_step(rootnode, actor_critic, critic):
    # Next node to expand
    node = find_next_node(rootnode)

    # Compute values on internal nodes
    if actor_critic is not None:
        pi_logits, val_logits = actor_critic(node.state[np.newaxis])
    else:
        assert critic is not None
        pi_logits = None
        val_logits = critic(node.state[np.newaxis])

    # Backprop value
    node.backprop(val_logits.item())

    # Don't need to calculate pi
    if node.terminal():
        return

    # Prior Pi
    if pi_logits is not None:
        pi = special.softmax(pi_logits.flatten())
        node.set_prior_pi(pi)
    else:
        node.make_children()
        next_nodes = node.get_child_nodes()
        tree.set_state_vals(critic, next_nodes)
        node.set_prior_pi(None)


def mct_search(go_env, num_searches, actor_critic=None, critic=None):
    # Setup the root
    root_group_map = go_env.get_canonical_group_map()
    rootstate = go_env.get_canonical_state()
    rootnode = tree.Node(rootstate, root_group_map)

    # The first iteration doesn't count towards the number of searches
    mct_step(rootnode, actor_critic, critic)

    # MCT Search
    for i in range(0, num_searches):
        mct_step(rootnode, actor_critic, critic)

    return rootnode
