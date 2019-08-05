import tensorflow as tf
import numpy as np
import time
import copy
import go_util

PI_CONST = None
U_CONST = None
TEMP_CONST = None


class Node:
    def __init__(self, parent, action_probs, state_value, board, move):
        '''
        Args:
            parent (?Node): parent Node
            action_probs (tensor): the policy action probs (flattened)
            state_value (float): the state value of this node
            board (GoEnv): the board, assuming already deep copied
            move (?tuple): the move that resulted in this node. None
                for root node
        '''
        self.parent = parent 
        # 1d array of the size that can hold the moves including pass, 
        # initially all None
        self.children = np.empty(board.board_width**2 + 1, dtype=object) 
        self.action_probs = action_probs # action probs of children
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



class MCTree:
    def __init__(self, board, forward_func, oppo_choose_move):
        '''
        Description:
            Construct a Monte Carlo Tree that has current board as root
        Args: 
            board (GoEnv): current board
            forward_func: function(GoEnv.state) => action_probs, state_value
            oppo_choose_move: function(GoEnv.state) => 1d move

        '''
        action_probs, state_value = forward_func(board.get_state())
        self.root = Node(None, action_probs, state_value, copy.deepcopy(board), None)
        self.forward_func = forward_func
        self.oppo_choose_move = oppo_choose_move
        self.board_width = board.board_width



    def select_best_child(self, node):
        '''
        Description: If it's our turn, select the child that 
            maximizes Q + U, where Q = V_sum / N, and 
            U = U_CONST * P / (1 + N), where P is action value.
            If it's oppo's turn, select the child using oppo_choose_move
            function. #TODO this could be inefficient if we visit this 
            white node mulitple times! consider merging the logic
        Args:
            node (Node): the parent node to choose from
        '''
        if U_CONST is None:
            raise Exception("U CONST is not set! (U = U_CONST * P / (1 + N))")
        # if it's our turn
        if node.board.get_next_player() == self.root.board.get_next_player():
            moves_2d = node.board.action_space
            moves_1d = list(map(go_util.action_2d_to_1d, moves_2d, [self.board_width] * len(moves_2d)))
            best_move = 0 # not None, because None is a valid move
            max_UCB = np.NINF # negative infinity
            # calculate Q + U for all children
            for move in moves_1d:
                if node.children[move] is None:
                    Q = 0
                    N = 0
                else:
                    Q = node.children[move].Q
                    N = node.N
                # get U for child
                U = node.action_probs[move] / (1 + N) * U_CONST
                # UCB: Upper confidence bound
                if Q + U > max_UCB:
                    max_UCB = Q + U
                    best_move = move
        # if it's opponent's turn
        else:
            # swap black and white channels
            best_move = self.oppo_choose_move(node.board.get_state()[[1, 0, 2, 3]])
        return node.children[best_move], best_move

    def expand(self, node, move):
        '''
        Description:
            Expand a new node from given node with the given move. If the
            give node is the end of a game, do nothing
        Args:
            node (Node): parent node to expand from
            move (1d): the move from parent to child
        '''
        if node.board.done:
            return None
        node.is_leaf = False
        child_board = copy.deepcopy(node.board)
        child_board.step(go_util.action_1d_to_2d(move, self.board_width))
        action_probs, state_value = self.forward_func(child_board.get_state())
        child = Node(node, action_probs, state_value, child_board, move)
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
