import unittest
import gym
import numpy as np
from go_ai.montecarlo import tree

class TestMCTS(unittest.TestCase):

    def setUp(self) -> None:
        self.env = gym.make('gym_go:go-v0', size=3, reward_method='real')
        self.action_length = self.env.size**2 + 1
        tree.U_CONST = 1
        tree.TEMP_CONST = 1

    def tearDown(self) -> None:
        self.env.close()

    def test_obvious_good_black_move(self):
        self.env.step(0)
        next_state, reward, done, info = self.env.step(None)

        def mock_forward_func(states):
            batch_size = states.shape[0]
            return np.zeros((batch_size,self.env.action_space)), np.zeros((batch_size,1))

        mct = tree.MCTree(next_state, mock_forward_func)
        action_probs = mct.get_action_probs(max_num_searches=0, temp=0)
        self.assertEqual(np.count_nonzero(action_probs[:-1]), 0, action_probs)
        self.assertEqual(action_probs[-1], 1, action_probs)

    def test_obvious_good_white_move(self):
        self.env.step(None)
        self.env.step(0)
        next_state, reward, done, info = self.env.step(None)

        def mock_forward_func(states):
            batch_size = states.shape[0]
            return np.zeros((batch_size, self.env.action_space)), np.zeros((batch_size, 1))

        mct = tree.MCTree(next_state, mock_forward_func)
        action_probs = mct.get_action_probs(max_num_searches=0, temp=0)
        self.assertEqual(np.count_nonzero(action_probs[:-1]), 0, action_probs)
        self.assertEqual(action_probs[-1], 1, action_probs)

if __name__ == '__main__':
    unittest.main()
