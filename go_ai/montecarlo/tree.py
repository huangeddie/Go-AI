import gym
import numpy as np

from go_ai import montecarlo
from go_ai.montecarlo.node import Node

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class MCTree:
    # Environment to call the stateless go logic APIs

    def __init__(self, val_func, state):
        """
        :param state: Starting state
        :param val_func: Takes in a batch of states and returns action
        probs and state values
        """
        canonical_state = GoGame.get_canonical_form(state)
        self.val_func = val_func

        canonical_state_val = val_func(canonical_state[np.newaxis])[0].item()

        self.root = Node(None, canonical_state_val, canonical_state)
        assert not self.root.visited()

        assert canonical_state.shape[1] == canonical_state.shape[2]
        self.board_size = canonical_state.shape[1]
        self.action_size = GoGame.get_action_size(self.root.state)

    def get_qvals(self, num_searches):
        '''
        Description:
            Select a child node that maximizes Q + U,
        Args:
            num_searches (int): maximum number of searches performed
            temp (number): temperature constant
        Returns:
            pi (1d np array): the search probabilities
            num_search (int): number of search performed
        '''
        rootstate = self.root.state

        num_search = 0
        if num_searches <= 0:
            # Avoid making nodes and stuff
            qvals, _ = montecarlo.qval_from_stateval(rootstate[np.newaxis], self.val_func)
            qvals = qvals[0]
        else:
            if not self.root.cached_children():
                self.cache_children(self.root)

            while num_search < num_searches:
                curr_node = self.root
                # keep going down the tree with the best move
                assert isinstance(curr_node, Node)
                while not curr_node.terminal and curr_node.visited():
                    curr_node, move = self.select_best_child(curr_node)
                if not curr_node.terminal:
                    curr_node, move = self.select_best_child(curr_node)
                if curr_node.height % 2 == 1 and not curr_node.terminal:
                    # We want to end on our turn
                    curr_node, move = self.select_best_child(curr_node)

                curr_node.parent.backup(curr_node.value)
                curr_node.visits += 1

                # increment search counter
                num_search += 1

            qvals = self.root.latest_qs()

        return qvals

    def select_best_child(self, node, u_const=1):
        """
        :param node:
        :param u_const: 'Exploration' factor of U
        :return: the child that
            maximizes Q + U, where Q = V_sum / N, and
            U = U_CONST * P / (1 + N), where P is action value.
            forward_func action probs
        """
        if not node.cached_children():
            self.cache_children(node)

        valid_moves = GoGame.get_valid_moves(node.state)
        invalid_values = (1 - valid_moves) * np.finfo(np.float).min

        Qs = node.latest_qs()
        prior_qs = node.prior_qs()
        prior_pi = montecarlo.exp_temp(prior_qs, 1, valid_moves)

        assert np.sum(prior_pi) > 0

        N = node.move_visits
        all_visits = np.sum(N)
        upper_confidence_bound = Qs + u_const * prior_pi * np.sqrt(all_visits) / (1 + N)
        best_move = np.argmax(upper_confidence_bound + invalid_values)

        return node.canon_children[best_move], best_move

    def cache_children(self, node):
        """
        Caches children for analysis by the forward function.
        Cached children have zero visit count, N = 0
        :param node:
        :return:
        """
        if node.terminal:
            return

        valid_move_idcs = GoGame.get_valid_moves(node.state)
        valid_move_idcs = np.argwhere(valid_move_idcs > 0).flatten()
        batch_qvals, batch_canonical_children = montecarlo.qval_from_stateval(node.state[np.newaxis], self.val_func)

        for idx, move in enumerate(valid_move_idcs):
            # Our qval is the negative state value of the canonical child
            Node((node, move), montecarlo.invert_qval(batch_qvals[0][move].item()), batch_canonical_children[idx])

    def step(self, action):
        '''
        Move the root down to a child with action. Throw away all nodes
        that are not in the child subtree. If such child doesn't exist yet,
        expand it.
        '''
        state = self.root.state
        if self.root.cached_children():
            canon_child = self.root.canon_children[action]
        else:
            childstate = GoGame.get_next_state(state, action)
            canonchildstate = GoGame.get_canonical_form(childstate)
            canon_child = Node(None, 0, canonchildstate) # State value doesn't matter

        self.root = canon_child

        assert isinstance(self.root, Node)
        self.root.parent = None
        self.root.update_height(0)

    def reset(self, state=None):
        if state is None:
            state = GoGame.get_init_board(self.board_size)

        self.__init__(self.val_func, state)

    def __str__(self):
        queue = [self.root]
        str_builder = ''
        while len(queue) > 0:
            curr_node = queue.pop(0)
            for child in curr_node.canon_children:
                if child is not None:
                    queue.append(child)
            str_builder += f'{curr_node}\n\n'

        return str_builder[:-2]
