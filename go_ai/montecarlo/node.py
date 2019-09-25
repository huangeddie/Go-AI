import numpy as np
from go_ai.montecarlo import GoGame

class Node:
    def __init__(self, parentaction, action_probs, state_value, state):
        '''
        Args:
            parent (?Node): parent Node
            action_probs (tensor): the policy action probs (flattened)
            state_value (float): the state value of this node
            state: state of the game as a numpy array
        '''
        if parentaction is not None:
            self.parent = parentaction[0]
            self.parent.children[parentaction[1]] = self
        else:
            self.parent = None

        self.height = self.parent.height + 1 if self.parent is not None else 0
        # 1d array of the size that can hold the moves including pass,
        # initially all None
        assert state.shape[1] == state.shape[2]
        board_size = state.shape[1]
        self.children = np.empty(board_size ** 2 + 1, dtype=object)
        self.action_probs = action_probs
        self.state = state
        self.turn = GoGame.get_turn(state)
        self.terminal = GoGame.get_game_ended(state)
        # number of time this node was visited
        self.N = 0
        self.V = state_value  # the evaluation of this node (value)
        self.Q_sum = 0

    def is_leaf(self):
        if self.N == 1:
            return True
        real_children = filter(lambda child: child is not None, self.children)
        return sum(map(lambda child: child.N, real_children)) <= 0

    def visited(self):
        return self.N > 0

    @property
    def cached_children(self):
        return (self.children != None).any()

    def avg_Q(self, move):
        child = self.children[move]
        avg_Q = (child.V.item() + child.Q_sum) / (1 + self.N)
        return -avg_Q

    def Qs(self):
        valid_moves = GoGame.get_valid_moves(self.state)
        Qs = [(self.avg_Q(move) if valid_moves[move] else 0) for move in range(GoGame.get_action_size(self.state))]
        return np.array(Qs)

    def back_propagate(self, value_incre):
        '''
        Description:
            Recursively increases the number visited by 1 and increase the
            V_sum by value increment from this node up to the root node.
        '''
        self.N += 1
        self.Q_sum += value_incre
        if self.parent is not None:
            self.parent.back_propagate(-value_incre)

    def __str__(self):
        return '{} {}H {}/{}VVS {}N'.format(np.sum(self.state[[0, 1]], axis=0), self.height, self.V, self.Q_sum, self.N)