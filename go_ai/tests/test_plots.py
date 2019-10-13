import random
import unittest
import time
import gym
import numpy as np

from go_ai import policies, metrics


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.random_policy = policies.RandomPolicy()
        self.greedy_policy = policies.MctPolicy('Greedy', policies.greedy_val_func, num_searches=0, temp=0)
        self.greedymct_pi = policies.MctPolicy('MCT', policies.greedy_val_func, num_searches=16, temp=0)
        self.go_env = gym.make('gym_go:go-v0', size=4)
        self.basedir = 'plots/'

        self.num_games = 128

    def test_starting_mct_search(self):
        state = self.go_env.reset()
        self.greedymct_pi.reset(state)
        self.greedymct_pi(state)
        metrics.plot_mct(self.greedymct_pi.tree, self.basedir + 'basic_mcts.png', max_layers=8, max_branch=8)

    def test_rand_mct_search(self):
        self.go_env.reset()
        for _ in range(random.randint(0, 8)):
            valid_moves = self.go_env.get_valid_moves()
            valid_moves[-1] = 0
            valid_move_idcs = np.argwhere(valid_moves > 0).flatten()
            a = np.random.choice(valid_move_idcs)
            self.go_env.step(a)
        state = self.go_env.get_canonical_state()
        self.greedymct_pi.reset(state)
        self.greedymct_pi(state)

        metrics.plot_mct(self.greedymct_pi.tree, self.basedir + 'rand_mcts.png', max_layers=8, max_branch=8)

    def test_symmetries(self):
        action = (1, 1)
        next_state, _, _, _ = self.go_env.step(action)
        metrics.plot_symmetries(next_state, self.basedir + 'symmetries.jpg')

    def test_traj(self):
        self.go_env.reset()
        t0 = time.time()
        metrics.plot_traj_fig(self.go_env, self.greedymct_pi, self.basedir + 'a_traj.png')
        t1 = time.time()
        print("Elapsed time:", t1 - t0)


if __name__ == '__main__':
    unittest.main()
