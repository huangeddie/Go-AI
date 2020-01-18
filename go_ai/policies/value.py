from queue import Queue

import numpy as np
import torch

from go_ai import search
from go_ai.models import pytorch_val_to_numpy
from go_ai.policies import Policy
from go_ai.search import mct


class Value(Policy):
    def __init__(self, name, val_func, mcts, temp=0):
        super(Value, self).__init__(name, temp)
        if isinstance(val_func, torch.nn.Module):
            self.pytorch_model = val_func
            val_func = pytorch_val_to_numpy(val_func)

        self.val_func = val_func
        self.mcts = mcts
        self.depth = int(np.log2(self.mcts)) if self.mcts > 0 else 0

    def __call__(self, go_env, **kwargs):
        """
        :param state: Unused variable since we already have the state stored in the tree
        :param step: Parameter used for getting the temperature
        :return:
        """
        rootnode = mct.val_search(go_env, self.mcts, self.val_func)
        qs = self.tree_to_qs(rootnode)

        exp_qs = np.exp(qs)
        pi = np.nansum(exp_qs, axis=0)
        pi = search.temp_norm(pi, self.temp, rootnode.valid_moves)

        if 'get_tree' in kwargs:
            if kwargs['get_tree']:
                return pi, rootnode
        else:
            return pi

    def tree_to_qs(self, rootnode):
        qs = np.full((self.depth + 1, rootnode.actionsize()), np.nan)

        # Iterate through root-child nodes, inner nodes
        queue = Queue()
        for child in rootnode.get_real_children():
            queue.put(child)

        while not queue.empty():
            node = queue.get()
            if node.level == 1:
                # Root child
                qs[0, node.first_action] = search.invert_vals(node.val)

            if not node.isleaf():
                # Inner node
                likely_node = min(node.get_real_children(), key=lambda node: node.val)
                level = likely_node.level

                orig_qval = qs[level - 1, node.first_action]
                if level % 2 == 0:
                    qval = np.nanmin([likely_node.val, orig_qval])
                else:
                    qval = np.nanmax([search.invert_vals(likely_node.val), orig_qval])

                qs[level - 1, node.first_action] = qval

                # Queue children
                for child in node.get_real_children():
                    queue.put(child)
        return qs

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"