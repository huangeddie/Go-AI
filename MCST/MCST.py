import tensorflow as tf
import numpy as np
import copy

class Node:
    def __init__(self, parent, action_values, state_value, board, move):
        '''
        Args:
            parent (?Node): parent Node
            action_values (tensor): the policy action values (flattened)
            state_value (float): the state value of this node
            board (GoEnv): the board, assuming already deep copied
            move (?tuple): the move that resulted in this node. None
                for root node
        '''
        self.parent = parent 
        # 1d array of the size that can hold the moves including pass, 
        # initially all None
        self.children = np.empty(board.board_size**2 + 1, dtype=object) 
        self.action_values = action_values # action values of children
        self.board = board 
        self.move = move 
        self.N = 1 # number of time this node was visited 
        self.V = state_value # the evaluation of this node (value)
        # the sum of values of all nodes in this subtree, including
        # this node
        self.V_sum = 0 
        self.increment_value_sum(self.V)
        # the total number of nodes in this subtree, including this node
        self.subtree_size = 0
        self.increment_subtree_size()
        self.is_leaf = True

    @property
    def Q(self):
        return self.V_sum / self.N

    def get_prior_for_child(self, move):
        if self.children[move] is None:
            return None
        return self.action_values[move]

    def increment_subtree_size(self):
        '''
        Description:
            Recursively increases the subtree size by 1 from this node up
            to the root node.
        '''
        self.subtree_size += 1
        if self.parent is not None:
            self.parent.increment_subtree_size()

    def increment_value_sum(self, increment):
        '''
        Description:
            Update the value sum of this node and all parent nodes, up 
            to root node
        '''
        self.V_sum += increment
        if self.parent is not None:
            self.parent.increment_value_sum(increment)

def select_best_child(node):
    '''
    Description: Select the child that maximizes Q + U
    '''
    2d_moves = node.board.action_space
    1d_moves = list(map(action_2d_to_1d, 2d_moves))
    best_move = 0 # not None, because None is a valid move
    max_UCB = np.NINF # negative infinity

    for move in 1d_moves:
        if node.children[move] is None:
            Q = 0
            N = 0
        else:
            Q = node.children[move].Q
            N = node.N
        # get U for child
        U = node.action_values[move] / (1 + N) * U_CONST
        # UCB: Upper confidence bound
        if Q + U > max_UCB:
            max_UCB = Q + U
            best_move = move

    # if haven't explored the best move yet, expand it
    if node.children[best_move] is None:
        child_board = copy.deepcopy(node.board)
        child_board.apply_move(best_move)
        # TODO use model to get action values and state value
        node.is_leaf = False
        child = Node(node, action_values[move], child_board, best_move)
        node.children[best_move] = child

    return node.children[best_move]


def action_2d_to_1d(action_2d, board_size):
    if action_2d is None:
        action_1d = board_size**2
    else:
        action_1d = action_2d[0] * board_size + action_2d[1]
    return action_1d


class MCTree:
    def __init__(self):
        '''
        Description:
            Construct a Monte Carlo Tree that has current board as root
        Args: 
        '''
        self.root = Node(None, )


    def select(self, node):
        '''
        Description:
            Select a child node that maximizes Q + U, where Q = V_sum / N,
            and U = UCB_CONST * P / (1 + N), where P is action value. 
        '''

