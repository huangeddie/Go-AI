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

        assert canonical_state.shape[1] == canonical_state.shape[2]
        self.board_size = canonical_state.shape[1]
        self.action_size = GoGame.get_action_size(self.root.state)
        self.rootgroup_map = None

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

        if num_searches <= 0:
            # Avoid making nodes and stuff
            qvals, _ = montecarlo.qs_from_stateval(rootstate[np.newaxis], self.val_func, [self.rootgroup_map])
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

        batch_qvals, batch_canonical_children = montecarlo.qs_from_stateval(node.state[np.newaxis], self.val_func)
        node.canon_children = np.empty(node.actionsize, dtype=object)

        for idx, move in enumerate(valid_move_idcs):
            # Child's state val is the inverted q value of the parent
            Node((node, move), montecarlo.invert_val(batch_qvals[0][move].item()), batch_canonical_children[idx])


    def visit_ancestry(self, node):
        curr_node = node
        curr_node.visits += 1
        while curr_node.parent is not None:
            parent = curr_node.parent

            actiontook = curr_node.actiontook
            parent.move_visits[actiontook] += 1
            parent.visits += 1

            curr_node = curr_node.parent

    def step(self, action):
        '''
        Move the root down to a child with action. Throw away all nodes
        that are not in the child subtree. If such child doesn't exist yet,
        expand it.
        '''
        state = self.root.state

        childstate, self.rootgroup_map = GoGame.get_next_state(state, action, self.rootgroup_map)
        canonchildstate = GoGame.get_canonical_form(childstate)
        canon_child = Node(None, 0, canonchildstate) # State value doesn't matter

        self.root = canon_child

        assert isinstance(self.root, Node)
        self.root.parent = None
        self.root.update_height(0)

    def reset(self, state=None):
        if state is None:
            state = GoGame.get_init_board(self.board_size)
        self.rootgroup_map = None
        self.__init__(self.val_func, state)

    def __str__(self):
        queue = [self.root]
        str_builder = ''
        while len(queue) > 0:
            curr_node = queue.pop(0)
            for child in curr_node.canon_children:
                if child is not None:
                    queue.append(child)
            str_builder += '{}\n\n'.format(curr_node)

        return str_builder[:-2]
