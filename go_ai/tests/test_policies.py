import unittest

import gym

from go_ai import game, policies


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        board_size = 5
        self.random_policy = policies.RandomPolicy()
        self.greedy_policy = policies.MctPolicy('Greedy', policies.greedy_val_func, num_searches=0, temp=0)
        self.greedy_mct_policy = policies.MctPolicy('MCTGreedy', policies.greedy_val_func, num_searches=board_size ** 2,
                                                    temp=0)
        self.go_env = gym.make('gym_go:go-v0', size=board_size)

        self.num_games = 128

    def test_rand_vs_greed(self):
        win_rate, _ = game.play_games(self.go_env, self.random_policy, self.greedy_policy, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greed_vs_rand(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_policy, self.random_policy, False, self.num_games)
        print(win_rate)
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
        print(win_rate)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_rand_vs_mct(self):
        win_rate, _ = game.play_games(self.go_env, self.random_policy, self.greedy_mct_policy, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_mct_vs_greed(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_mct_policy, self.greedy_policy, False, self.num_games)
        print(win_rate)
        self.assertGreaterEqual(win_rate, 0.8)

    def test_greed_vs_mct(self):
        win_rate, _ = game.play_games(self.go_env, self.greedy_policy, self.greedy_mct_policy, False, self.num_games)
        print(win_rate)
        self.assertLessEqual(win_rate, 0.2)


if __name__ == '__main__':
    unittest.main()
