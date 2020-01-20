from queue import Queue

import numpy as np
import torch

from go_ai import search
from go_ai.models import pytorch_val_to_numpy
from go_ai.policies import Policy
from go_ai.search import mct


class Value(Policy):
    def __init__(self, name, val_func, args=None):
        super(Value, self).__init__(name, args.temp if args is not None else 0)
        if isinstance(val_func, torch.nn.Module):
            self.pytorch_model = val_func
            val_func = pytorch_val_to_numpy(val_func)

        self.val_func = val_func
        self.width = args.width if args is not None else 0
        self.depth = args.depth if args is not None else 0
        self.gamma = args.gamma if args is not None else 0

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        if 'debug' in kwargs:
            debug =  kwargs['debug']
        else:
            debug = False

        qs, rootnode = mct.val_search(go_env, self.width, self.depth, self.val_func, debug)

        exp_qs = np.nan_to_num(np.exp(qs))
        gammas = self.gamma ** np.arange(self.depth + 1)
        pi = np.matmul(exp_qs.T, gammas)
        pi = search.temp_norm(pi, self.temp, rootnode.valid_moves())

        if debug:
            return pi, qs, rootnode
        else:
            return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.width}/{self.depth}S {self.temp:.2f}T]-{self.name}"