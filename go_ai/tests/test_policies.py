import unittest

import gym

from go_ai import game, policies

random_policy = policies.RandomPolicy()
greedy_policy = policies.QTempPolicy('Greedy', policies.greedy_val_func, temp=0)
# mct_policy = policies.MctPolicy('MCT', 5, policies.greedy_val_func, temp=0, num_searches=10)
go_env = gym.make('gym_go:go-v0', size=5)

num_games = 128


class MyTestCase(unittest.TestCase):
    def test_rand_vs_greed(self):
        go_env.reset()
        win_rate, _ = game.play_games(go_env, random_policy, greedy_policy, False, num_games)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greed_vs_rand(self):
        go_env.reset()
        win_rate, _ = game.play_games(go_env, random_policy, greedy_policy, False, num_games)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    # def test_mct_vs_rand(self):
    #     go_env.reset()
    #     win_rate, _ = game.play_games(go_env, mct_policy, random_policy, False, num_games)
    #     self.assertAlmostEqual(win_rate, 1, delta=0.1)
    #
    # def test_rand_vs_mct(self):
    #     go_env.reset()
    #     win_rate, _ = game.play_games(go_env, random_policy, mct_policy, False, num_games)
    #     self.assertAlmostEqual(win_rate, 1, delta=0.1)
    #
    # def test_mct_vs_greed(self):
    #     go_env.reset()
    #     win_rate, _ = game.play_games(go_env, mct_policy, greedy_policy, False, num_games)
    #     self.assertGreaterEqual(win_rate, 0.5)
    #
    # def test_greed_vs_mct(self):
    #     go_env.reset()
    #     win_rate, _ = game.play_games(go_env, greedy_policy, mct_policy, False, num_games)
    #     self.assertLessEqual(win_rate, 0.5)


if __name__ == '__main__':
    unittest.main()
