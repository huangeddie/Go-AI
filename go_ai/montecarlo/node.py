import numpy as np

from go_ai import montecarlo
from go_ai.montecarlo import GoGame


class Node:
    def __init__(self, parentaction, prior_value, state):
        '''
        Args:
            parent (?Node): parent Node
            prior_value (?float): the state value of this node
            state: state of the game as a numpy array
        '''
        if parentaction is not None:
            self.parent = parentaction[0]
            self.parent.canon_children[parentaction[1]] = self
            self.actiontook = parentaction[1]
        else:
            self.actiontook = None
            self.parent = None

        self.height = self.parent.height + 1 if self.parent is not None else 0
        assert len(state.shape) == 3, (state, state.shape)
        assert state.shape[1] == state.shape[2], (state, state.shape)

        self.state = state
        self.prior_value = prior_value  # the value of this node prior to lookahead
        self.terminal = GoGame.get_game_ended(state)

        # Visits
        self.actionsize = GoGame.get_action_size(state)
        self.canon_children = None
        self.move_visits = np.zeros(self.actionsize)
        self.visits = 0

    def update_height(self, new_height):
        self.height = new_height
        children_height = self.height + 1

        if not self.is_leaf():
            for child in self.canon_children:
                if isinstance(child, Node):
                    child.update_height(children_height)

    def latest_value(self):
        if self.is_leaf():
            return self.prior_value
        else:
            qs = []
            for child in self.canon_children:
                if child is None:
                    qs.append(0)
                else:
                    qs.append(montecarlo.invert_val(child.latest_value()))
            max_qval = max(qs)

            if self.prior_value is None:
                return max_qval
            # Average of prior of max q-val
            return (self.prior_value + max_qval) / 2

    def is_leaf(self):
        return self.canon_children is None

    def latest_q(self, move):
        """
        Assumes child exists
        :param move:
        :return:
        """
        return montecarlo.invert_val(self.canon_children[move].latest_value())

    def latest_qs(self):
        qs = []
        for child in self.canon_children:
            if child is None:
                qs.append(0)
            else:
                qs.append(montecarlo.invert_val(child.latest_value()))

        return np.array(qs)

    def __str__(self):
        return '{} {}H {}V {}N'.format(np.sum(self.state[[0, 1]], axis=0), self.height, self.prior_value,
                                       np.sum(self.move_visits))
