import unittest

import gym

from go_ai import game, policies
from go_ai.models import value
import utils
import torch


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        board_size = 5
        self.greedy_mct_policy = policies.MCTS('MCTGreedy', policies.greedy_val_func, num_searches=8,
                                               temp=0)

        self.go_env = gym.make('gym_go:go-v0', size=board_size)

        self.num_games = 256

    def test_mctval_vs_val(self):
        self.go_env = gym.make('gym_go:go-v0', size=9)
        curr_model = value.ValueNet(9)
        curr_model.load_state_dict(torch.load('../../bin/checkpoint.pt'))

        mct_pi = policies.MCTS('MCT', curr_model, num_searches=100, temp=0.03125, temp_steps=24)
        val_pi = policies.MCTS('MCT', curr_model, num_searches=0, temp=0.03125, temp_steps=24)

        win_rate, _, _ = game.play_games(self.go_env, mct_pi, val_pi, False, self.num_games)
        print(win_rate)
        self.assertGreaterEqual(win_rate, 0.6)



    def test_mct_vs_greed(self):
        win_rate, _, _ = game.play_games(self.go_env, self.greedy_mct_policy, policies.GREEDY_PI, False, self.num_games)
        print(win_rate)
        self.assertGreaterEqual(win_rate, 0.6)


    def test_greed_vs_rand(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.GREEDY_PI, policies.RAND_PI, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_mctcheck_vs_check(self):
        args = utils.hyperparameters()
        checkpoint_model = value.ValueNet(9, 9)
        checkpoint_model.load_state_dict(torch.load(args.checkpath))
        check = policies.MCTS('Checkpoint', checkpoint_model, 0, args.temp, args.tempsteps)
        mct_check = policies.MCTS('Checkpoint', checkpoint_model, 4, args.temp, args.tempsteps)

        self.go_env = gym.make('gym_go:go-v0', size=9)
        win_rate, _, _ = game.play_games(self.go_env, mct_check, check, False, self.num_games)
        print(win_rate)
        self.assertGreaterEqual(win_rate, 0.6)

    def test_greed_vs_greed(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.GREEDY_PI, policies.GREEDY_PI, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 0.5, delta=0.1)

    def test_smartgreed_vs_greed(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.SMART_GREEDY_PI, policies.GREEDY_PI, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 0.5, delta=0.1)

    def test_rand_vs_greed(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.RAND_PI, policies.GREEDY_PI, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greedy_and_basegreedymct_are_equal(self):
        self.greedy_mct_policy.num_searches = 0
        done = False
        state = self.go_env.get_canonical_state()
        while not done:
            greedy_pi = policies.GREEDY_PI(state, None)
            mct_pi = self.greedy_mct_policy(state, None)
            self.assertTrue((greedy_pi == mct_pi).all(), (state[:2], greedy_pi, mct_pi))
            a = self.go_env.uniform_random_action()
            _, _, done, _ = self.go_env.step(a)
            self.greedy_mct_policy.step(a)
            state = self.go_env.get_canonical_state()

    def test_mct_vs_rand(self):
        win_rate, _, _ = game.play_games(self.go_env, self.greedy_mct_policy, policies.RAND_PI, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_rand_vs_mct(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.RAND_PI, self.greedy_mct_policy, False, self.num_games)
        print(win_rate)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)

    def test_greed_vs_mct(self):
        win_rate, _, _ = game.play_games(self.go_env, policies.GREEDY_PI, self.greedy_mct_policy, False, self.num_games)
        print(win_rate)
        self.assertLessEqual(win_rate, 0.2)


if __name__ == '__main__':
    unittest.main()
