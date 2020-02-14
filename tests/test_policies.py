import unittest
import os

from go_ai import utils


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if 'tests' in os.getcwd():
            os.chdir('../')
        self.num_games = 256

    def test_checkpoint(self):
        args1 = utils.hyperparameters(['--size=5', '--depth=0', '--temp=0.1', '--customdir=bin/checkpoints/2020-01-20/'])
        args2 = utils.hyperparameters(['--size=5', '--depth=0', '--temp=0.1', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_mctval_vs_val(self):
        args1 = utils.hyperparameters(['--size=7', '--mcts=256', '--temp=1', '--baseline'])
        args2 = utils.hyperparameters(['--size=7', '--mcts=0', '--temp=0.05', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_val_vs_ac(self):
        args1 = utils.hyperparameters(['--size=9', '--model=val', '--depth=0', '--temp=0.05', '--baseline'])
        args2 = utils.hyperparameters(['--size=9', '--model=ac', '--mcts=0', '--temp=1', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_mctac_vs_ac(self):
        args1 = utils.hyperparameters(['--size=9', '--model=ac', '--mcts=81', '--baseline'])
        args2 = utils.hyperparameters(['--size=9', '--model=ac', '--mcts=0', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_smartgreed_vs_greed(self):
        args1 = utils.hyperparameters(['--size=5', '--model=smartgreedy', '--depth=0', '--temp=0.1', '--baseline'])
        args2 = utils.hyperparameters(['--size=5', '--model=greedy', '--depth=0', '--temp=0.1', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_mct_vs_greed(self):
        args1 = utils.hyperparameters(['--size=5', '--model=greedy', '--depth=1', '--temp=0.1', '--baseline'])
        args2 = utils.hyperparameters(['--size=5', '--model=greedy', '--depth=0', '--temp=0.1', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_mct_vs_rand(self):
        args1 = utils.hyperparameters(['--size=5', '--model=greedy', '--mcts=10', '--temp=1', '--baseline'])
        args2 = utils.hyperparameters(['--size=5', '--model=rand'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_greed_vs_rand(self):
        args1 = utils.hyperparameters(['--size=5', '--model=greedy', '--mcts=0', '--temp=0.1', '--baseline'])
        args2 = utils.hyperparameters(['--size=5', '--model=rand'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertGreaterEqual(win_rate, 0.6)

    def test_greed_vs_greed(self):
        args1 = utils.hyperparameters(['--size=5', '--model=greedy', '--depth=0', '--temp=0.1', '--baseline'])
        args2 = utils.hyperparameters(['--size=5', '--model=greedy', '--depth=0', '--temp=0.1', '--baseline'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertAlmostEqual(win_rate, 0.5, delta=0.1)

    def test_rand_vs_rand(self):
        args1 = utils.hyperparameters(['--size=5', '--model=rand'])
        args2 = utils.hyperparameters(['--size=5', '--model=rand'])

        win_rate, _, _ = utils.multi_proc_play(args1, args2, self.num_games, workers=8)

        self.assertAlmostEqual(win_rate, 0.5, delta=0.1)


if __name__ == '__main__':
    unittest.main()
