import numpy as np

from go_ai import search, data
from go_ai.policies import Policy
from go_ai.search import mct, tree


class ActorCritic(Policy):
    def __init__(self, name, model, args=None):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(ActorCritic, self).__init__(name, temp=args.temp)
        self.pt_model = model
        self.ac_func = model.create_numpy('actor_critic')
        self.val_func = model.create_numpy('critic')
        self.pi_func = model.create_numpy('actor')
        self.mcts = args.mcts

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """

        if self.mcts > 0:
            rootnode = mct.mct_search(go_env, self.mcts, actor_critic=self.ac_func)
            qs = self.tree_to_qs(rootnode)

            # Raise to temperature
            pi = (qs[1] ** (1 / self.temp))
            pi = pi / np.sum(pi)

            # Noise to guarantee all moves may be explored
            valid_moves = go_env.get_valid_moves()
            noise = valid_moves / valid_moves.sum()
            pi = 0.98 * pi + 0.02 * noise

            assert np.allclose(np.sum(pi), 1), np.sum(pi)

        elif self.mcts == 0:
            # Just use value function to get policy
            rootnode = mct.mct_search(go_env, self.mcts, critic=self.val_func)
            q_logits = rootnode.inverted_children_values()
            pi = search.temp_norm(np.exp(q_logits), self.temp, rootnode.valid_moves())
            qs = [q_logits]
        else:
            # Just use policy function and don't search
            assert self.mcts < 0
            # Get tree node for debugging purposes
            rootnode = tree.Node(go_env.state, go_env.group_map)
            state = go_env.get_canonical_state()
            policy_scores = self.pi_func(state[np.newaxis])
            policy_scores = policy_scores[0]
            valid_moves = data.GoGame.get_valid_moves(state)
            pi = search.temp_softmax(policy_scores, self.temp, valid_moves)
            qs = [pi]

        if 'debug' in kwargs:
            debug = kwargs['debug']
        else:
            debug = False
        if debug:
            return pi, qs, rootnode

        return pi

    def tree_to_qs(self, rootnode):
        qs = np.empty((2, rootnode.actionsize()))
        qs[0] = rootnode.prior_pi
        qs[1] = rootnode.get_visit_counts()

        return qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
