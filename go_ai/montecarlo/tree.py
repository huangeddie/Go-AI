from sklearn.preprocessing import normalize
import numpy as np

from go_ai.montecarlo import GoGame, piqval_from_actorcritic
from go_ai.montecarlo.node import Node


class MCTree:
    # Environment to call the stateless go logic APIs

    def __init__(self, forward_func, state, save_orig_root=False):
        """
        :param state: Starting state
        :param forward_func: Takes in a batch of states and returns action
        probs and state values
        :param save_orig_root: Save the original root Node for debugging
        """
        self.forward_func = forward_func

        action_probs, state_value = forward_func(state[np.newaxis])
        action_probs, state_value = action_probs[0], state_value[0].item()

        self.root = Node(None, action_probs, state_value, state)
        assert not self.root.visited()
        self.root.back_propagate(self.root.V)

        assert state.shape[1] == state.shape[2]
        self.board_size = state.shape[1]
        self.action_size = GoGame.get_action_size(self.root.state)

        self.save_orig_root = save_orig_root
        if save_orig_root:
            self.orig_root = self.root

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
        num_search = 0
        if max_num_searches <= 0:
            rootstate = self.root.state
            pis, _ = self.forward_func(rootstate[np.newaxis])
            pi = pis[0]
            assert len(pi.shape) == 1
            return pi
        else:
            while num_search < max_num_searches:
                curr_node = self.root
                # keep going down the tree with the best move
                while curr_node.visited() and not curr_node.terminal:
                    curr_node, move = self.select_best_child(curr_node)

                curr_node.back_propagate(curr_node.V)

                # increment search counter
                num_search += 1

            qvals = list(map(lambda node: node.N if node is not None else 0, self.root.children))
            qvals = np.array(qvals)

        if temp > 0:
            pi = normalize([qvals ** (1 / temp)], norm='l1')[0]
        else:
            best_actions = (qvals == np.max(qvals))
            pi = normalize(best_actions[np.newaxis], norm='l1')[0]

        return pi

    def select_best_child(self, node, u_const=1):
        """
        :param node:
        :param u_const: 'Exploration' factor of U
        :return: the child that
            maximizes Q + U, where Q = V_sum / N, and
            U = U_CONST * P / (1 + N), where P is action value.
            forward_func action probs
        """
        if not node.cached_children:
            self.cache_children(node)

        valid_moves = GoGame.get_valid_moves(node.state)
        valid_move_idcs = np.argwhere(valid_moves > 0).flatten()
        best_move = None
        max_UCB = np.NINF  # negative infinity
        # calculate Q + U for all children
        for move in valid_move_idcs:
            Q = node.avg_Q(move)
            child = node.children[move]
            Nsa = child.N
            Psa = node.action_probs[move]
            U = u_const * Psa * np.sqrt(node.N) / (1 + Nsa)

            # UCB: Upper confidence bound
            if Q + U > max_UCB:
                max_UCB = Q + U
                best_move = move

        if best_move is None:
            raise Exception("MCTS: move shouldn't be None, please debug")

        return node.children[best_move], best_move

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

        batch_pis, batch_qvals, batch_canonical_children = piqval_from_actorcritic(node.state[np.newaxis],
                                                                                   self.forward_func)
        for idx, move in enumerate(valid_move_idcs):
            Node((node, move), batch_pis[0][idx], batch_qvals[0][idx], batch_canonical_children[idx])

    def step(self, action):
        '''
        Move the root down to a child with action. Throw away all nodes
        that are not in the child subtree. If such child doesn't exist yet,
        expand it.
        '''
        child = self.root.children[action]
        if child is None:
            next_state = GoGame.get_next_state(self.root.state, action)
            next_turn = GoGame.get_turn(next_state)
            canonical_state = GoGame.get_canonical_form(next_state, next_turn)
            action_probs, state_value = self.forward_func(canonical_state[np.newaxis])
            action_probs, state_value = action_probs[0], state_value[0].item()
            # Set parent to None because we know we're going to set it as root
            child = Node(None, action_probs, state_value, next_state)

        self.root = child
        self.root.parent = None
        if not self.root.visited():
            self.root.back_propagate(self.root.V)

    def reset(self, state=None):
        if state is None:
            state = GoGame.get_init_board(self.board_size)
        self.__init__(self.forward_func, state, self.save_orig_root)

    def save_root(self):
        self.save_orig_root = True
        self.orig_root = self.root

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
