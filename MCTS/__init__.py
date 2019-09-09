from sklearn.preprocessing import normalize
import numpy as np
import copy
from go_ai import go_utils

class Node:
    def __init__(self, parent, action_probs, state_value, board):
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
        self.children = np.empty(board.size**2 + 1, dtype=object)
        self.action_probs = action_probs
        self.board = board
        # number of time this node was visited
        self.N = 1
        self.V = state_value # the evaluation of this node (value)
        self.V_sum = self.V # the sum of values of this subtree

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
    def __init__(self, board, forward_func):
        '''
        Description:
            Construct a Monte Carlo Tree that has current board as root
        Args:
            board (GoEnv): current board
            forward_func: function(GoEnv.state) => action_probs, state_value
        '''
        action_probs, state_value = forward_func(board.get_state())
        self.root = Node(None, action_probs, state_value, copy.deepcopy(board))
        self.forward_func = forward_func
        self.board_size = board.size
        self.our_player = board.turn

    def select_best_child(self, node, u_const=1):
        '''
        Description: If it's our turn, select the child that
            maximizes Q + U, where Q = V_sum / N, and
            U = U_CONST * P / (1 + N), where P is action value.
            If it's oppo's turn, randomly select the child according to
            forward_func action probs.
        Args:
            node (Node): the parent node to choose from
        '''

        # if it's our turn
        if node.board.turn == self.our_player:
            moves_1d = np.arange(node.board.action_space)
            best_move = None
            max_UCB = np.NINF # negative infinity
            # calculate Q + U for all children
            for move in moves_1d:
                if node.children[move] is None:
                    Q = 0
                    N = 0
                else:
                    child = node.children[move]
                    Q = child.Q
                    N = child.N
                # get U for child
                U = node.action_probs[move] * np.sqrt(node.N) / (1 + N) * u_const
                # UCB: Upper confidence bound
                if Q + U > max_UCB:
                    max_UCB = Q + U
                    best_move = move
        # if it's opponent's turn, choose an action based on action prob
        else:
            best_move = go_utils.random_weighted_action([node.action_probs])

        if best_move is None:
            raise Exception("MCTS: move shouldn't be None, please debug")
        return node.children[best_move], best_move

    def expand(self, node, move):
        '''
        Description:
            Expand a new node from given node with the given move.
        Args:
            node (Node): parent node to expand from
            move (1d): the move from parent to child
        Returns:
            If a node is created, return the new node. When the game ends
            with the node passed in, nothing is created and return the node
        '''
        # if we reach a end state, return this node
        if node.board.game_ended:
            return node
        child_board = copy.deepcopy(node.board)
        state, _, _, info = child_board.step(go_utils.action_1d_to_2d(move, self.board_size))
        # if it's our move, save our action prob
        canonical_state = child_board.get_canonical_state()
        action_probs, state_value = self.forward_func(canonical_state)
        child = Node(node, action_probs, state_value, child_board)
        node.children[move] = child
        return child


    def perform_search(self, max_num_searches=100, temp=1):
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

        num_search = 0

        while num_search < max_num_searches:
            # keep going down the tree with the best move
            curr_node = self.root
            next_node, move = self.select_best_child(curr_node)
            while next_node is not None:
                curr_node = next_node
                next_node, move = self.select_best_child(curr_node)
            # reach a leaf and expand
            leaf = self.expand(curr_node, move)
            curr_node.back_propagate(leaf.V)
            # increment counters
            num_search += 1

        N = []
        for child in self.root.children:
            if child is None:
                N.append(0)
            else:
                N.append(child.N)
        N = np.array(N)
        pi = normalize([N ** (1 / temp)], norm='l1')[0]

        return pi, num_search
