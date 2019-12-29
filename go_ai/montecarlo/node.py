import numpy as np

from go_ai import montecarlo
from go_ai.montecarlo import GoGame

class Node:
    def __init__(self, state, group_map, parent=None):
        '''
        Args:
            parent (?Node): parent Node
            prior_value (?float): the state value of this node
            state: state of the game as a numpy array
        '''

        self.state = state
        self.group_map = group_map
        self.terminal = GoGame.get_game_ended(state)
        self.winning = GoGame.get_winning(state)
        self.valid_moves = GoGame.get_valid_moves(state)

        # Visits
        self.actionsize = GoGame.get_action_size(state)
        self.canon_children = np.empty(self.actionsize, dtype=object)
        self.visits = 0

        self.prior_pi = None
        self.post_vals = []
        self.parent = parent
        if parent is None:
            self.height = 0
        else:
            self.height = self.parent.height + 1

    def traverse(self, move):
        child = self.canon_children[move]
        if child is not None:
            return child
        else:
            batch_moves = np.array([move])
            next_states, next_gmps = GoGame.get_batch_next_states(self.state, batch_moves, self.group_map, canonical=True)
            child = Node(next_states[0], next_gmps[0], self)
            self.canon_children[move] = child
            return child

    def backprop(self, val):
        if val is not None:
            self.post_vals.append(val)
        self.visits += 1
        if self.parent is not None:
            inverted_val = montecarlo.invert_vals(val) if val is not None else None
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
        for i in range(self.actionsize):
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
        avg_q = np.mean(self.post_vals)
        return f'{self.visits}N {self.post_vals[0]:.2f}V {avg_q:.2f}AV {self.height}H'
