import numpy as np

from go_ai import search
from go_ai.search import GoGame


def set_state_vals(val_func, nodes):
    states = list(map(lambda node: node.state, nodes))
    vals = val_func(np.array(states))
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
        self.terminal = GoGame.get_game_ended(state)
        self.winning = GoGame.get_winning(state)
        self.valid_moves = GoGame.get_valid_moves(state)

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

    # =================
    # Basic Tree API
    # =================
    def isleaf(self):
        # Not the same as whether the state is terminal or not
        return (self.canon_children == None).all()

    def isroot(self):
        return self.parent is None

    def make_child(self, action, state, groupmap):
        child_node = Node(state, groupmap, self)
        self.canon_children[action] = child_node
        if child_node.level == 1:
            child_node.first_action = action
        else:
            assert self.first_action is not None
            child_node.first_action = self.first_action
        return child_node

    def make_children(self):
        children, child_gmps = GoGame.get_children(self.state, self.group_map, canonical=True)
        actions = np.argwhere(self.valid_moves).flatten()
        assert len(actions) == len(children)
        for action, state, groupmap in zip(actions, children, child_gmps):
            self.make_child(action, state, groupmap)

        return self.get_real_children()

    def get_real_children(self):
        return list(filter(lambda node: node is not None, self.canon_children))

    def actionsize(self):
        return GoGame.get_action_size(self.state)

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
        for i in range(self.actionsize()):
            if not self.valid_moves[i]:
                ucbs.append(np.finfo(np.float).min)
            else:
                prior_q = self.prior_pi[i]
                child = self.canon_children[i]
                if child is not None:
                    n = child.visits
                    avg_q = np.mean(child.post_vals)
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
