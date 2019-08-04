import tensorflow as tf
import numpy as np
import time
import copy

PI_CONST = None
U_CONST = None
TEMP_CONST = None


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
        # number of time this node was visited, which is the same as
        # subtree size
        self.N = 1 
        self.V = state_value # the evaluation of this node (value)
        self.V_sum = self.V # the sum of values of this subtree
        if parent is not None:
            parent.back_propagate(self.V)
        self.is_leaf = True

    @property
    def Q(self):
        return self.V_sum / self.N 

    def back_propagate(self, value_incre):
        '''
        Description:
            Recursively increases the number visited by 1 and increase the 
            V_sum by value increment from this node up to the root node.
        '''
        self.N += 1
        self.V_sum += value_incre
        if self.parent is not None:
            self.parent.back_propagate(value_incre)


def action_2d_to_1d(action_2d, board_size):
    if action_2d is None:
        action_1d = board_size**2
    else:
        action_1d = action_2d[0] * board_size + action_2d[1]
    return action_1d


class MCTree:
    def __init__(self, board, forward_func):
        '''
        Description:
            Construct a Monte Carlo Tree that has current board as root
        Args: 
            board (GoEnv): current board
            forward_func (function(GoEnv) => action_values, state_value)
        '''
        action_values, state_value = forward_func(board)
        self.root = Node(None, action_values, state_value, copy.deepcopy(board), None)
        self.forward_func = forward_func


    def select_best_child(node):
        '''
        Description: Select the child that maximizes Q + U, 
            where Q = V_sum / N, and U = U_CONST * P / (1 + N),
            where P is action value.
        Args:
            node (Node): the parent node to choose from
        '''
        if U_CONST is None:
            raise Exception("U CONST is not set! (U = U_CONST * P / (1 + N))")
        moves_2d = node.board.action_space
        moves_1d = list(map(action_2d_to_1d, moves_2d))
        best_move = 0 # not None, because None is a valid move
        max_UCB = np.NINF # negative infinity

        for move in moves_1d:
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

        return node.children[best_move], best_move


    def expand(self, node, move):
        '''
        Description:
            Expand a new node from given node with the given move
        Args:
            node (Node): parent node to expand from
            move (1d): the move from parent to child
        '''
        node.is_leaf = False
        child_board = copy.deepcopy(node.board)
        child_board.step(move)
        action_values, state_value = self.forward_func(child_board)
        child = Node(node, action_values, state_value, child_board, move)
        node.children[move] = child
        return child


    def perform_search(self, max_num_searches=1000, max_time=300):
        '''
        Description:
            Select a child node that maximizes Q + U,  
        Args:
            max_num_searches (int): maximum number of searches performed
            max_time (float): maxtime spend in this function in seconds
        Returns:
            pi (1d np array): the search probabilities
            num_search (int): number of search performed
            time_spent (float): number of seconds spent
        '''
        if PI_CONST is None:
            raise Exception("PI_CONST is not set! (pi = PI_CONST * N**(1 / TEMP_CONST))")
        if TEMP_CONST is None:
            raise Exception("TEMP_CONST is not set! (pi = PI_CONST * N**(1 / TEMP_CONST))")

        start_time = time.time()
        num_search = 0
        time_spent = 0

        while num_search < max_num_searches and time_spent < max_time:
            # keep going down the tree with the best move
            curr_node = self.root
            next_node, move = self.select_best_child(curr_node)
            while next_node is not None:
                curr_node = next_node
                next_node, move = self.select_best_child(curr_node)
            # reach a leaf and expand
            self.expand(curr_node, move)
            # increment counters
            num_search += 1
            time_spent = time.time() - start_time

        N = []
        for child in self.root.children:
            if child is None:
                N.append(0)
            else:
                N.append(child.N)
        N = np.array(N)
        pi = PI_CONST * N ** (1 / TEMP_CONST)

        return pi, num_search, time_spent
