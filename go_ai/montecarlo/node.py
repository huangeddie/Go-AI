import numpy as np

from go_ai.montecarlo import GoGame


class Node:
    def __init__(self, parentaction, state_value, state):
        '''
        Args:
            parent (?Node): parent Node
            state_value (float): the state value of this node
            state: state of the game as a numpy array
        '''
        if parentaction is not None:
            self.parent = parentaction[0]
            self.parent.canon_children[parentaction[1]] = self
            self.lastaction = parentaction[1]
        else:
            self.lastaction = None
            self.parent = None

        self.height = self.parent.height + 1 if self.parent is not None else 0
        # 1d array of the size that can hold the moves including pass,
        # initially all None
        assert len(state.shape) == 3, (state, state.shape)
        assert state.shape[1] == state.shape[2], (state, state.shape)
        board_size = state.shape[1]
        self.canon_children = np.empty(board_size ** 2 + 1, dtype=object)
        self.state = state
        self.terminal = GoGame.get_game_ended(state)
        # number of time this node was visited
        self.value = state_value  # the evaluation of this node (value)
        action_size = GoGame.get_action_size(state)
        self.post_qsums = np.zeros(action_size)
        self.move_visits = np.zeros(action_size)


    def visited(self):
        return np.sum(self.move_visits) > 0

    @property
    def cached_children(self):
        return (self.canon_children != None).any()

    def prior_q(self, move):
        canon_child = self.canon_children[move]
        return 1 - canon_child.value

    def prior_qs(self):
        valid_moves = GoGame.get_valid_moves(self.state)
        Qs = [(self.prior_q(move) if valid_moves[move] else 0) for move in range(GoGame.get_action_size(self.state))]
        return np.array(Qs)

    def latest_q(self, move):
        visits = self.move_visits[move]
        if visits <= 0:
            return self.prior_q(move)
        else:
            return self.post_qsums[move] / visits

    def latest_qs(self):
        valid_moves = GoGame.get_valid_moves(self.state)
        children = self.canon_children
        Qs = []
        for move in range(GoGame.get_action_size(self.state)):
            if not valid_moves[move]:
                # Invalid move (0)
                Qs.append(0)
                continue
            Qs.append(self.latest_q(move))

        return np.array(Qs)

    def back_propagate(self, newq, move):
        '''
        Description:
            Recursively increases the number visited by 1 and increase the
            q value increment from this node up to the root node.
        '''
        self.post_qsums[move] += newq
        self.move_visits[move] += 1
        if self.parent is not None:
            self.parent.back_propagate(1 - newq, self.lastaction)

    def __str__(self):
        return '{} {}H {}V {}N'.format(np.sum(self.state[[0, 1]], axis=0), self.height, self.value,
                                       np.sum(self.move_visits))
