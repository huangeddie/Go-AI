import unittest

import gym

from go_ai import game, policies



class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.random_policy = policies.RandomPolicy()
        self.greedy_policy = policies.QTempPolicy('Greedy', policies.greedy_val_func, temp=0)
        self.greedy_mct_policy = policies.MctPolicy('MCT', 4, policies.greedy_val_func, temp=0, num_searches=16)
        self.go_env = gym.make('gym_go:go-v0', size=4)

        self.num_games = 128


    def test_rand_vs_greed(self):
        win_rate, _ = game.play_games(self.go_env, self.random_policy, self.greedy_policy, False, self.num_games)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greed_vs_rand(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_policy, self.random_policy, False, self.num_games)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_greedy_and_basegreedymct_are_equal(self):
        self.greedy_mct_policy.num_searches = 0
        done = False
        state = self.go_env.get_canonical_state()
        while not done:
            greedy_pi = self.greedy_policy(state, None)
            mct_pi = self.greedy_mct_policy(state, None)
            self.assertTrue((greedy_pi == mct_pi).all(), (state[:2], greedy_pi, mct_pi))
            a = self.go_env.uniform_random_action()
            _, _, done, _ = self.go_env.step(a)
            self.greedy_mct_policy.step(a)
            state = self.go_env.get_canonical_state()

    def test_mct_vs_rand(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_mct_policy, self.random_policy, False, self.num_games)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_rand_vs_mct(self):
        win_rate, _ = game.play_games(self.go_env, self.random_policy, self.greedy_mct_policy, False, self.num_games)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_mct_vs_greed(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_mct_policy, self.greedy_policy, False, self.num_games)
        self.assertGreaterEqual(win_rate, 0.8)

    def test_greed_vs_mct(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_policy, self.greedy_mct_policy, False, self.num_games)
        self.assertLessEqual(win_rate, 0.2)


if __name__ == '__main__':
    unittest.main()
