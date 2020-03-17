import numpy as np
import torch

from go_ai import search
from go_ai.policies import Policy
from go_ai.search import mct


class Value(Policy):
    def __init__(self, name, val_func, args=None):
        super(Value, self).__init__(name, args.temp if args is not None else 0)
        if isinstance(val_func, torch.nn.Module):
            self.pt_model = val_func
            val_func = self.pt_model.val_numpy

        self.val_func = val_func
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

        rootnode = mct.val_search(go_env, self.mcts, self.val_func, debug)
        if self.mcts > 0:
            qs = rootnode.get_visit_counts()
        else:
            q_logits = rootnode.get_q_logits()
            qs = np.exp(q_logits)
        pi = search.temp_norm(qs, self.temp, rootnode.valid_moves())

        if debug:
            qs = rootnode.get_q_logits()
            return pi, [qs, qs], rootnode
        else:
            rootnode.destroy()
            del rootnode
            return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
