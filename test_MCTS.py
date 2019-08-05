import unittest
import gym
import numpy as np
import MCTS 
import go_util

class TestMCTS(unittest.TestCase):

    def setUp(self) -> None:
        self.env = gym.make('gym_go:go-v0', size='S', reward_method='real')
        self.action_length = self.env.board_width**2 + 1
        MCTS.PI_CONST = 1
        MCTS.U_CONST = 1
        MCTS.TEMP_CONST = 1

    def tearDown(self) -> None:
        self.env.close()

    def test_one_step_search(self):
        '''
        This tests a simple one step search
        _,_,_    B,_,_                      1,0,0
        _,_,_ => _,_,_     should return    0,0,0
        _,_,_    _,_,_                      0,0,0
        '''
        def mock_forward_func(state):
            '''
            Empty board:
                action probs: 1 at the top left corner (0,0)
                state value: 0 (50% chance of wining)
            After (0,0) is black and nothing else:
                action probs: 1 at (0, 1)
                state value: 1
            Otherwise:
                action probs: NaN
                state value: -1
            '''
            action_probs = np.zeros(self.action_length)
            # empty board
            if np.count_nonzero(state[0:3]) == 0:
                action_probs[0] = 1
                state_value = 0
            # black at (0,0)
            elif state[0,0,0] == 1 and np.count_nonzero(state[0]) == 1 \
                and np.count_nonzero(state[1]) == 0:
                    idx = go_util.action_2d_to_1d((0, 1), self.env.board_width)
                    action_probs[idx] = 1
                    state_value = 1
            # unexpected states
            else:
                raise Exception("Unexpected state")
            return action_probs, state_value

        tree = MCTS.MCTree(self.env, mock_forward_func, None)
        # only perform one search, should only reach 1 state
        pi, num_search, time_spent = tree.perform_search(1)
        # the first move should have pi = 1
        self.assertEqual(pi[0], 1)
        # the rest of the moves should have pi = 0
        for i in range(1, self.action_length):
            self.assertEqual(pi[i], 0)
        self.assertEqual(num_search, 1)

    def test_two_step_search(self):
        '''
        This tests a simple two-step search
        _,_,_    B,_,_    B,W,_                     2,0,0
        _,_,_ => _,_,_ => _,_,_    should return    0,0,0
        _,_,_    _,_,_    _,_,_                     0,0,0
        '''
        def mock_forward_func(state):
            '''
            Empty board:
                action probs: 1 at the top left corner (0,0)
                state value: 0 (50% chance of wining)
            After (0,0) is black and nothing else:
                action probs: 1 at (0, 1)
                state value: 1
            When (0,0) is black, (0,1) is white:
                action probs: 1 at (1, 0)
                state value: 0.5
            Otherwise:
                raise exception
            '''
            action_probs = np.zeros(self.action_length)
            # empty board
            if np.count_nonzero(state[0:3]) == 0:
                action_probs[0] = 1
                state_value = 0
            # black at (0,0)
            elif state[0,0,0] == 1 and np.count_nonzero(state[0]) == 1 \
                and np.count_nonzero(state[1]) == 0:
                    idx = go_util.action_2d_to_1d((0, 1), self.env.board_width)
                    action_probs[idx] = 1
                    state_value = 1
            # black at (0,0), white at (0,1)
            elif state[0,0,0] == 1 and np.count_nonzero(state[0]) == 1 \
                and state[1,0,1] == 1 and np.count_nonzero(state[1]) == 1:
                    idx = go_util.action_2d_to_1d((1, 0), self.env.board_width)
                    action_probs[idx] = 1
                    state_value = 0.5
            # unexpected states
            else:
                raise Exception("Unexpected state")
            return action_probs, state_value
        
        def mock_oppo(state):
            '''
            Black at (0,0): return (0,1)
            Else: raise exception
            '''
            # black at (0,0)
            if state[1,0,0] == 1 and np.count_nonzero(state[1]) == 1 \
                and np.count_nonzero(state[0]) == 0:
                    return go_util.action_2d_to_1d((0, 1), self.env.board_width)
            raise Exception("Unexpected state for opponent")

        tree = MCTS.MCTree(self.env, mock_forward_func, mock_oppo)
        pi, num_search, time_spent = tree.perform_search(2)
        # the first move should have pi = 2
        self.assertEqual(pi[0], 2)
        # the rest of the moves should have pi = 0
        for i in range(1, self.action_length):
            self.assertEqual(pi[i], 0)

        # test V_sum, V and N to be expected values
        self.assertEqual(tree.root.V_sum, 1.5)
        self.assertEqual(tree.root.V, 0)
        self.assertEqual(tree.root.N, 3)

        black_node = tree.root.children[0]
        self.assertEqual(black_node.V_sum, 1.5)
        self.assertEqual(black_node.V, 1)
        self.assertEqual(black_node.N, 2)

        white_node = black_node.children[go_util.action_2d_to_1d((0, 1), self.env.board_width)]
        self.assertEqual(white_node.V_sum, 0.5)
        self.assertEqual(white_node.V, 0.5)
        self.assertEqual(white_node.N, 1)

    def test_two_branch_search(self):
        '''
        This tests a two branch search, search 3 times
        .5,.5,__    B,_,_   
        __,__,__ => _,_,_   value = 0
        __,__,__    _,_,_   

                \   _,B,_                                  1,2,0
                 \> _,_,_   value = 1     should return    0,0,0
                    _,_,_                                  0,0,0
        '''
        move_01 = go_util.action_2d_to_1d((0, 1), self.env.board_width)
        move_10 = go_util.action_2d_to_1d((1, 0), self.env.board_width)
        def mock_forward_func(state):
            '''
            Empty board:
                action probs: 0.5 for (0,0) and (0,1)
                state value: 0 
            After (0,0) is black and nothing else:
                action probs: 1 at (0, 1) # shouldn't matter
                state value: 0
            When (0,1) is black and nothing else:
                action probs: 1 at (1, 0) # shouldn't matter
                state value: 1
            Otherwise:
                raise exception
            '''
            action_probs = np.zeros(self.action_length)
            # empty board
            if np.count_nonzero(state[0:3]) == 0:
                action_probs[0] = 0.5
                action_probs[move_01] = 0.5
                state_value = 0
            # black at (0,0)
            elif state[0,0,0] == 1 and np.count_nonzero(state[0]) == 1 \
                and np.count_nonzero(state[1]) == 0:
                    action_probs[move_10] = 1
                    state_value = 0
            # black at (0,1)
            elif state[0,0,1] == 1 and np.count_nonzero(state[0]) == 1 \
                and np.count_nonzero(state[1]) == 0:
                    action_probs[move_01] = 1
                    state_value = 1
            # black at (0,1) and white at (0,0), for leaf node
            elif state[0,0,1] == 1 and np.count_nonzero(state[0]) == 1 \
                and state[1,0,0] == 1 and np.count_nonzero(state[1]) == 1:
                    state_value = 0
            # unexpected states
            else:
                raise Exception("Unexpected state")
            return action_probs, state_value
        
        def mock_oppo(state):
            '''
            Black at (0,1): return (0,0)
            Else: raise exception
            '''
            # black at (0,1)
            if state[1,0,1] == 1 and np.count_nonzero(state[1]) == 1 \
                and np.count_nonzero(state[0]) == 0:
                    return 0 # move (0,0) in 1d
            raise Exception("Unexpected state for opponent")

        tree = MCTS.MCTree(self.env, mock_forward_func, mock_oppo)
        pi, num_search, time_spent = tree.perform_search(3)
        # check pi
        self.assertEqual(pi[0], 1)
        self.assertEqual(pi[move_01], 2)
        # the rest of the moves should have pi = 0
        for i in range(2, self.action_length):
            self.assertEqual(pi[i], 0)

    def test_end_of_game(self):
        '''
        Tests that two player pass
        '''
        move_pass = go_util.action_2d_to_1d(None, self.env.board_width)
        def mock_forward_func(state):
            '''
            Empty board:
                action probs: 1 for pass
                state value: 0 
            Otherwise:
                raise exception
            '''
            action_probs = np.zeros(self.action_length)
            # empty board
            if np.count_nonzero(state[0:3]) == 0:
                action_probs[move_pass] = 1
                state_value = 0
            # unexpected states
            else:
                raise Exception("Unexpected state")
            return action_probs, state_value
        
        def mock_oppo(state):
            '''
            Empty board: return pass
            Else: raise exception
            '''
            # empty board
            if np.count_nonzero(state[0:3]) == 0:
                return move_pass
            raise Exception("Unexpected state for opponent")

        tree = MCTS.MCTree(self.env, mock_forward_func, mock_oppo)
        pi, num_search, time_spent = tree.perform_search(3)
        # check pi
        self.assertEqual(pi[move_pass], 3)
        # the rest of the moves should have pi = 0
        for i in range(1, self.action_length):
            self.assertEqual(pi[i], 0)


if __name__ == '__main__':
    unittest.main()
