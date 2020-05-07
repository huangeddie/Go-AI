import numpy as np

from go_ai import models
from go_ai import search
from go_ai.policies import Policy
from go_ai.search import tree


class QVal(Policy):
    def __init__(self, name, engine, args=None):
        super().__init__(name, args.temp if args is not None else 0)
        if isinstance(engine, models.RLNet):
            self.pt_model = engine
            self.q_func = self.pt_model.create_numpy('critic')
        else:
            self.q_func = engine
        self.mcts = args.mcts if args is not None else 0

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        state = go_env.get_canonical_state()
        valid_moves = go_env.get_valid_moves()

        new_shape = np.array(state.shape)
        new_shape[0] = 7

        state_action = np.zeros(new_shape)
        state_action[:-1] = state

        qvals = np.zeros(go_env.action_space.n)
        for a, valid in enumerate(valid_moves):
            if valid:
                state_action[-1] = 0
                if a < go_env.action_space.n - 1:
                    r, c = a // go_env.size, a % go_env.size
                    state_action[-1, r, c] = 1
                qval_logit = self.q_func(state_action[np.newaxis])
                qvals[a] = qval_logit.item()

        pi = search.temp_softmax(qvals, self.temp, valid_moves)

        debug = False
        if 'debug' in kwargs:
            debug = kwargs['debug']

        if debug:
            rootnode = tree.Node(go_env.state, go_env.group_map)
            return pi, [qvals], rootnode
        return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
