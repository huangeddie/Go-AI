import numpy as np
from scipy import special

from go_ai import search
from go_ai.search import GoGame

def get_state_vals(val_func, nodes):
    states = list(map(lambda node: node.state, nodes))
    vals = val_func(np.array(states))
    return vals

def set_state_vals(val_func, nodes):
    vals = get_state_vals(val_func, nodes)
    for val, node in zip(vals, nodes):
        node.set_value(val.item())

    return vals


class Node:
    def __init__(self, state, group_map, parent=None):
        '''
        Args:
            parent (?Node): parent Node
            prior_value (?float): the state value of this node
            state: state of the game as a numpy array
        '''

        # Go
        self.state = state
        self.group_map = group_map

        # Links
        self.parent = parent
        self.canon_children = np.empty(self.actionsize(), dtype=object)

        # Level
        if parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

        # Value
        self.val = None
        self.first_action = None

        # MCT
        self.visits = 0
        self.prior_pi = None
        self.post_vals = []

    def destroy(self):
        for child in self.canon_children:
            if child is not None:
                child.destroy()
        del self.state
        del self.group_map
        del self.parent
        del self.canon_children

    # =================
    # Basic Tree API
    # =================
    def terminal(self):
        return GoGame.get_game_ended(self.state)

    def winning(self):
        return GoGame.get_winning(self.state)

    def isleaf(self):
        # Not the same as whether the state is terminal or not
        return (self.canon_children == None).all()

    def isroot(self):
        return self.parent is None

    def make_child(self, action, state, group_map):
        child_node = Node(state, group_map, self)
        self.canon_children[action] = child_node
        if child_node.level == 1:
            child_node.first_action = action
        else:
            assert self.first_action is not None
            child_node.first_action = self.first_action
        return child_node

    def make_children(self):
        children, child_gmps = GoGame.get_children(self.state, self.group_map, canonical=True)
        actions = np.argwhere(self.valid_moves()).flatten()
        assert len(actions) == len(children)
        for action, state, group_map in zip(actions, children, child_gmps):
            self.make_child(action, state, group_map)

        return self.get_real_children()

    def get_real_children(self):
        return list(filter(lambda node: node is not None, self.canon_children))

    def actionsize(self):
        return GoGame.get_action_size(self.state)

    def valid_moves(self):
        return GoGame.get_valid_moves(self.state)

    def step(self, move):
        child = self.canon_children[move]
        if child is not None:
            return child
        else:
            next_state, next_gmp = GoGame.get_next_state(self.state, move, self.group_map, canonical=True)
            child = self.make_child(move, next_state, next_gmp)
            return child

    # =====================
    # Value
    # =====================
    def set_value(self, val):
        self.val = val

    def get_value(self):
        return self.val

    def set_val_prior(self):
        valid_moves = self.valid_moves()
        where_valid = np.argwhere(valid_moves).flatten()
        q_logits = self.get_q_logits()
        self.prior_pi[where_valid] = special.softmax(q_logits[where_valid])

        assert not np.isnan(self.prior_pi).any()

    def get_q_logits(self):
        self.prior_pi = np.zeros(self.actionsize())

        q_logits = []
        for child in self.canon_children:
            if child is not None:
                q_logits.append(search.invert_vals(child.val))
            else:
                q_logits.append(0)
        return np.array(q_logits)

    # =====================
    # MCT API
    # =====================

    def backprop(self, val):
        if val is not None:
            self.post_vals.append(val)
        self.visits += 1
        if self.parent is not None:
            inverted_val = search.invert_vals(val) if val is not None else None
            self.parent.backprop(inverted_val)

    def set_prior_pi(self, prior_pi):
        self.prior_pi = prior_pi

    def get_visit_counts(self):
        move_visits = []
        for child in self.canon_children:
            if child is None:
                move_visits.append(0)
            else:
                move_visits.append(child.visits)
        return np.array(move_visits)

    def get_ucbs(self):
        ucbs = []
        valid_moves = self.valid_moves()
        for i in range(self.actionsize()):
            if not valid_moves[i]:
                ucbs.append(np.finfo(np.float).min)
            else:
                prior_q = self.prior_pi[i]
                child = self.canon_children[i]
                if child is not None:
                    n = child.visits
                    if len(child.post_vals) <= 0:
                        avg_q = search.invert_vals(np.tanh(child.get_value))
                    else:
                        avg_q = search.invert_vals(np.mean(np.tanh(child.post_vals)))
                else:
                    n = 0
                    avg_q = 0

                u = 1.5 * prior_q * np.sqrt(self.visits) / (1 + n)

                ucbs.append(avg_q + u)
        return ucbs

    def __str__(self):
        result = ''
        if self.val is not None:
            result += f'{self.val:.2f}V'
        if len(self.post_vals) > 0:
            result += f' {np.mean(self.post_vals):.2f}AV'

        result += f' {self.level}L {self.visits}N'

        return result
