import numpy as np

from go_ai import models
from go_ai import search
from go_ai.policies import Policy
from go_ai.search import mct


class Value(Policy):
    def __init__(self, name, engine, args=None):
        super(Value, self).__init__(name, args.temp if args is not None else 0)
        if isinstance(engine, models.RLNet):
            self.pt_model = engine
            self.val_func = self.pt_model.create_numpy('critic')
        else:
            self.val_func = engine
        self.mcts = args.mcts if args is not None else 0

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        if 'debug' in kwargs:
            debug = kwargs['debug']
        else:
            debug = False

        rootnode = mct.mct_search(go_env, self.mcts, critic=self.val_func)
        if self.mcts > 0:
            qs = rootnode.get_visit_counts()
            assert np.sum(qs) > 0, rootnode
        else:
            q_logits = rootnode.inverted_children_values()
            qs = np.exp(q_logits)
        pi = search.temp_norm(qs, self.temp, rootnode.valid_moves())

        if debug:
            qs = rootnode.inverted_children_values()
            return pi, [qs, qs], rootnode
        else:
            rootnode.destroy()
            del rootnode
            return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
