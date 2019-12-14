import gym
import numpy as np

from go_ai import montecarlo
from go_ai.montecarlo.node import Node

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class MCTree:

    def get_qvals(self, go_env, num_searches, val_func, pi_func):
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

        if num_searches <= 0:
            # Avoid making nodes and stuff
            qvals, _ = montecarlo.qs_from_valfunc(rootstate, self.val_func, self.rootgroup_map)
            qvals = qvals[0]
        else:
            if self.root.is_leaf():
                self.make_children(self.root)

            for _ in range(num_searches):
                curr_node = self.root
                assert isinstance(curr_node, Node)
                while not curr_node.is_leaf():
                    curr_node, move = self.select_best_child(curr_node)

                assert curr_node.is_leaf()
                if not curr_node.terminal:
                    self.make_children(curr_node)

                self.visit_ancestry(curr_node)

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
        assert not node.is_leaf()

        valid_moves = GoGame.get_valid_moves(node.state)
        invalid_values = (1 - valid_moves) * np.finfo(np.float).min

        Qs = node.latest_qs()

        N = node.move_visits
        all_visits = np.sum(N)
        upper_confidence_bound = Qs + u_const * np.sqrt(all_visits) / (1 + N)
        best_move = np.argmax(upper_confidence_bound + invalid_values)

        return node.canon_children[best_move], best_move

    def make_children(self, node):
        """
        Makes children for analysis by the forward function.
        :param node:
        :return:
        """
        if node.terminal:
            return

        valid_move_idcs = GoGame.get_valid_moves(node.state)
        valid_move_idcs = np.argwhere(valid_move_idcs > 0).flatten()

        qvals, canonical_children = montecarlo.qs_from_valfunc(node.state, self.val_func)
        node.canon_children = np.empty(node.actionsize, dtype=object)

        for idx, move in enumerate(valid_move_idcs):
            # Child's state val is the inverted q value of the parent
            Node((node, move), montecarlo.invert_val(qvals[move].item()), canonical_children[idx])


    def visit_ancestry(self, node):
        curr_node = node
        curr_node.visits += 1
        while curr_node.parent is not None:
            parent = curr_node.parent

            actiontook = curr_node.actiontook
            parent.move_visits[actiontook] += 1
            parent.visits += 1

            curr_node = curr_node.parent