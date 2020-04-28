import unittest
import os
import numpy as np

from go_ai import utils
import gym


class PolicyVersusPolicy(unittest.TestCase):
    def setUp(self) -> None:
        if 'tests' in os.getcwd():
            os.chdir('../')
        self.num_games = 256
        self.env = gym.make('gym_go:go-v0', size=5)

    def set_obvious_move_state(self):
        self.env.reset()
        self.env.step(0)  # Black
        self.env.step(1)  # White
        self.env.step(2)  # Black
        self.env.step(None)  # White

    def test_attn_obvious_move(self):
        args = utils.hyperparameters(['--size=5', '--model=attn', '--mct=-1'])

        self.set_obvious_move_state()

        # Obvious move is to pass and win the game
        policy, _ = utils.baselines.create_policy(args)
        pi = policy(self.env)
        self.assertEqual(pi[-1], 1)
        self.assertTrue(np.allclose(pi[:-1], 0))

    def test_greedy_obvious_move(self):
        args = utils.hyperparameters(['--size=5', '--model=greedy'])

        self.set_obvious_move_state()

        # Obvious move is to pass and win the game
        policy, _ = utils.baselines.create_policy(args)
        pi = policy(self.env)
        self.assertEqual(pi[-1], 1)
        self.assertTrue(np.allclose(pi[:-1], 0))



if __name__ == '__main__':
    unittest.main()
