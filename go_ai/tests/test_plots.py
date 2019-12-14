import random
import unittest
import time
import gym
import numpy as np

from go_ai import policies, measurements


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.random_policy = policies.Random()
        self.greedy_policy = policies.MCTS('Greedy', policies.greedy_val_func, num_searches=0, temp=0)
        self.greedymct_pi = policies.MCTS('MCT', policies.greedy_val_func, num_searches=4, temp=0)
        self.go_env = gym.make('gym_go:go-v0', size=4)
        self.basedir = 'plots/'

        self.num_games = 128

    def test_starting_mct_search(self):
        state = self.go_env.reset()
        self.greedymct_pi.reset(state)
        self.greedymct_pi(state, 0)
        measurements.plot_mct(self.greedymct_pi.tree, self.basedir + 'basic_mcts.pdf', max_layers=8, max_branch=8)

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
        self.greedymct_pi(state, 0)

        measurements.plot_mct(self.greedymct_pi.tree, self.basedir + 'rand_mcts.pdf', max_layers=8, max_branch=8)

    def test_symmetries(self):
        action = (1, 1)
        next_state, _, _, _ = self.go_env.step(action)
        measurements.plot_symmetries(next_state, self.basedir + 'symmetries.pdf')

    def test_traj(self):
        self.go_env.reset()
        t0 = time.time()
        measurements.plot_traj_fig(self.go_env, self.greedymct_pi, self.basedir + 'a_traj.pdf')
        t1 = time.time()
        print("Elapsed time:", t1 - t0)


if __name__ == '__main__':
    unittest.main()
