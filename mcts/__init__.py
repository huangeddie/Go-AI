from sklearn.preprocessing import normalize
import numpy as np
import gym

GoGame = gym.make('gym_go:go-v0', size=0).gogame

class Node:
    def __init__(self, parent, action_probs, state_value, state):
        '''
        Args:
            parent (?Node): parent Node
            action_probs (tensor): the policy action probs (flattened)
            state_value (float): the state value of this node
            state: state of the game as a numpy array
        '''
        self.parent = parent
        self.height = parent.height + 1 if parent is not None else 0
        # 1d array of the size that can hold the moves including pass,
        # initially all None
        assert state.shape[1] == state.shape[2]
        board_size = state.shape[1]
        self.children = np.empty(board_size**2 + 1, dtype=object)
        self.action_probs = action_probs
        self.state = state
        self.turn = GoGame.get_turn(state)
        # number of time this node was visited
        self.N = 1
        self.V = state_value # the evaluation of this node (value)
        self.V_sum = self.V # the sum of values of this subtree
        # later when soft reset is_expanded will be false
        self.is_expanded = True

    @property
    def Q(self):
        if self.N == 0:
            return 0
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

    def soft_reset(self):
        self.is_expanded = False
        self.N = 0
        self.V_sum = 0
        for child in self.children:
            if child is not None:
                child.soft_reset()

    def __str__(self):
        return '{} {}H {}/{}VVS {}N'.format(np.sum(self.state[[0,1]], axis=0), self.height, self.V, self.V_sum, self.N)

class MCTree:
    # Environment ot call the stateless go logic APIs

    def __init__(self, state, forward_func):
        '''
        Description:
            Construct a Monte Carlo Tree that has current board as root
        Args:
            state (GoEnv): current board
            forward_func: function(GoEnv.state) => action_probs, state_value
        '''
        action_probs, state_value = forward_func(state[np.newaxis])
        action_probs, state_value = action_probs[0], state_value[0]

        self.root = Node(None, action_probs, state_value, state)
        self.forward_func = forward_func
        assert state.shape[1] == state.shape[2]
        self.board_size = state.shape[1]
        self.action_size = GoGame.get_action_size(self.root.state)
        self.our_player = GoGame.get_turn(state)

    def get_action_probs(self, max_num_searches, temp):
        '''
        Description:
            Select a child node that maximizes Q + U,
        Args:
            max_num_searches (int): maximum number of searches performed
            temp (number): temperature constant
        Returns:
            pi (1d np array): the search probabilities
            num_search (int): number of search performed
        '''

        if max_num_searches <= 0:
            valid_moves = GoGame.get_valid_moves(self.root.state)
            states = []
            for move, valid in enumerate(valid_moves):
                if valid > 0:
                    state = GoGame.get_next_state(self.root.state, move)
                    states.append(state)
            state_batch = np.array(states)
            _, vals = self.forward_func(state_batch)

            action_probs = np.zeros(self.action_size)
            curr_idx = 0
            for move in range(self.action_size):
                if valid_moves[move]:
                    action_probs[move] = vals[curr_idx].numpy()
                    curr_idx += 1

            if temp > 0:
                action_probs += np.min(action_probs)
                action_probs = normalize((action_probs**(1/temp))[np.newaxis], norm='l1')[0]
            else:
                best_action = np.argmax(action_probs)
                action_probs = np.zeros(self.action_size)
                action_probs[best_action] = 1

            return action_probs, 0
        else:
            num_search = 0
            while num_search < max_num_searches:
                # keep going down the tree with the best move
                curr_node = self.root
                next_node, move = self.select_best_child(curr_node)
                while next_node is not None and next_node.is_expanded:
                    curr_node = next_node
                    next_node, move = self.select_best_child(curr_node)
                # reach a leaf and expand
                leaf = self.expand(curr_node, move)
                curr_node.back_propagate(leaf.V)
                # increment counters
                num_search += 1

            N = list(map(lambda node: node.N if node is not None else 0, self.root.children))
            N = np.array(N)
            if temp > 0:
                pi = normalize([N ** (1 / temp)], norm='l1')[0]
            else:
                bestA = np.argmax(N)
                pi = np.zeros(len(N))
                pi[bestA] = 1

            return pi, num_search

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

        moves_1d = np.arange(GoGame.get_action_size(node.state))
        valid_moves = GoGame.get_valid_moves(node.state)
        best_move = None
        max_UCB = np.NINF  # negative infinity
        # calculate Q + U for all children
        for move in moves_1d:
            if node.children[move] is None:
                if valid_moves[move] > 0:
                    Q = 0
                    N = 0
                else:
                    Q = np.NINF
                    N = np.NINF
            else:
                child = node.children[move]
                Q = child.Q if node.turn == self.our_player else -child.Q
                N = child.N
            # get U for child
            U = node.action_probs[move] * np.sqrt(node.N) / (1 + N) * u_const
            # UCB: Upper confidence bound
            if Q + U > max_UCB:
                max_UCB = Q + U
                best_move = move

        if best_move is None:
            raise Exception("MCTS: move shouldn't be None, please debug")

        return node.children[best_move], best_move


    def expand(self, node, move):
        '''
        Description:
            Expand a new node from given node with the given move. Or after
            soft reset, set the child corresponding to the move to be expanded
        Args:
            node (Node): parent node to expand from
            move (1d): the move from parent to child
        Returns:
            If a node is created, return the new node. When the game ends
            with the node passed in, nothing is created and return the node
        '''
        # if we reach a end state, return this node
        if GoGame.get_game_ended(node.state):
            return node
        # if the child node already exists, but not expanded
        if node.children[move] is not None:
            child = node.children[move]
            child.is_expanded = True
            child.N = 1
            child.V_sum = child.V
        # if the child node doesn't exist, create it
        else:
            next_state = GoGame.get_next_state(node.state, move)
            next_turn = GoGame.get_turn(next_state)
            # save action prob and value
            canonical_state = GoGame.get_canonical_form(next_state, next_turn)
            action_probs, state_value = self.forward_func(canonical_state[np.newaxis])
            action_probs, state_value = action_probs[0], state_value[0]
            if next_turn != self.our_player:
                state_value = -state_value
            child = Node(node, action_probs, state_value, next_state)
            node.children[move] = child
        return child


    def step(self, action):
        '''
        Move the root down to a child with action. Throw away all nodes
        that are not in the child subtree. If such child doesn't exist yet,
        expand it.
        '''
        child = self.root.children[action]
        if child is None:
            child = self.expand(self.root, action)

        self.root = child
        self.root.parent = None

        self.root.N = 1
        self.root.V_sum = self.root.V
        for child in self.root.children:
            if child is not None:
                child.soft_reset()

    def __str__(self):
        queue = [self.root]
        str_builder = ''
        while len(queue) > 0:
            curr_node = queue.pop(0)
            for child in curr_node.children:
                if child is not None:
                    queue.append(child)
            str_builder += '{}\n\n'.format(curr_node)

        return str_builder[:-2]