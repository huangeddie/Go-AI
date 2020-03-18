import gym
import numpy as np

from go_ai import search
from go_ai.policies import Policy
from go_ai.search import mct

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class Attn(Policy):
    def __init__(self, name, model, args=None):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(Attn, self).__init__(name, temp=args.temp)
        self.pt_model = model
        self.ac_func = model.ac_numpy
        self.val_func = model.val_numpy
        self.mcts = args.mcts

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """

        if self.mcts > 0:
            rootnode = mct.ac_search(go_env, self.mcts, self.ac_func)
            qs = self.tree_to_qs(rootnode)

            pi = qs[1] ** (1 / self.temp)
            pi = pi / np.sum(pi)

        elif self.mcts == 0:
            rootnode = mct.val_search(go_env, self.mcts, self.val_func)
            q_logits = rootnode.get_q_logits()
            qs = np.exp(q_logits)
            pi = search.temp_norm(qs, self.temp, rootnode.valid_moves())
        else:
            assert self.mcts < 0
            rootnode = mct.val_search(go_env, self.mcts, self.val_func)
            state = go_env.get_canonical_state()
            children = go_env.get_children(canonical=True, padded=True)
            policy_scores, _ = self.ac_func(state[np.newaxis], children)
            policy_scores = policy_scores[0]
            valid_moves = GoGame.get_valid_moves(state)
            pi = search.temp_softmax(policy_scores, self.temp, valid_moves)

        if 'debug' in kwargs:
            debug = kwargs['debug']
        else:
            debug = False
        if debug:
            qs = self.tree_to_qs(rootnode)
            return pi, qs, rootnode

        return pi

    def tree_to_qs(self, rootnode):
        qs = np.empty((2, rootnode.actionsize()))
        qs[0] = rootnode.prior_pi
        qs[1] = rootnode.get_visit_counts()

        return qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
