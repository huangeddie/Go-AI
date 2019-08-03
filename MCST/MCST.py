import tensorflow as tf
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


def expand_leaf(parent, action_values, state_value, move):
    '''
    Description:
        Create the child node corresponding to the move for the parent 
        passed in. 
    Args:
        parent (Node): the parent to expand
        action_values: action values of the child to add
        state_value: the state value of the child to add
    '''
    # update this parent's attributes
    parent.is_leaf = False
    
    # TODO fix this function after writing select
    board = copy.deepcopy(parent.board)
    board.apply_move(move)
    child = Node(parent, action_values[move], board, move)
    parent.children[move] = child


def select_best_child(node):
    '''
    Description: Select the child that maximizes Q + U
    '''
    legal_moves = 


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

