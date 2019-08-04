import unittest
import gym
import numpy as np
import . as MCST

class TestMCST(unittest.TestCase):

    def setUp(self) -> None:
        self.env = gym.make('gym_go:go-v0', size='S', reward_method='real')


    def tearDown(self) -> None:
        self.env.close()

    def test_tree_construct(self):
        tree = MCSTree(self.env, self.mock_forward_func)
        self.assertEqual(actual, expected)

        def mock_forward_func(self, board):
            '''
            Empty board:
                action values: 1 at the top left corner
                state value: 0
            '''
            action_values = np.zeros((board.board_size, board.board_size))
            action_values[0, 0] = 1
            state_value = 1
            return action_values, state_value

