import gym
import numpy as np

from go_ai import search, models
from go_ai.policies import Policy
from go_ai.search import mct

GoGame = gym.make('gym_go:go-v0', size=0).gogame


class Attn(Policy):
    def __init__(self, name, model: models.RLNet, args=None):
        """
        :param branches: The number of actions explored by actor at each node.
        :param depth: The number of steps to explore with actor. Includes opponent,
        i.e. even depth means the last step explores the opponent's
        """
        super(Attn, self).__init__(name, temp=args.temp)
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

        rootnode = mct.val_search(go_env, self.mcts, self.val_func)
        state = go_env.get_canonical_state()
        policy_scores = self.pi_func(state[np.newaxis])
        policy_scores = policy_scores[0]
        valid_moves = GoGame.get_valid_moves(state)
        pi = search.temp_softmax(policy_scores, self.temp, valid_moves)

        if 'debug' in kwargs:
            debug = kwargs['debug']
        else:
            debug = False
        if debug:
            return pi, [policy_scores], rootnode

        return pi

    def __str__(self):
        return f"{self.__class__.__name__}[{self.mcts}S {self.temp:.2f}T]-{self.name}"
