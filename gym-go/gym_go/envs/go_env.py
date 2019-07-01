import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


# The side length of each board size
BOARD_SIZES = {
    'S': 7,
    'M': 13,
    'L': 19,
}


class GoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size='S', reward_method='heuristic'):
        '''
        @param reward_method: either 'heuristic' or 'real', 'heuristic' 
            gives # your pieces - # oppo pieces. 'real' gives 0 for in-game
            move, 1 for winning, -1 for losing, 0 for draw.
        '''
        # determine board size
        try:
            self.board_width = BOARD_SIZES[size]
        except KeyError as e:
            raise Exception('Board size should be one of {}'.format(list(BOARD_SIZES.keys())))
            
        # create board
        # access: [Y, O, K, P][Row number][Column number]
        self.board = np.zeros((4, self.board_width, self.board_width), dtype=np.int)
        
        self.reward_method = reward_method
        

    def print_state(self):
        print("Your pieces:")
        print(self.board[0])
        print("Opponent's pieces:")
        print(self.board[1])
        print("Ko-protection:")
        print(self.board[2])
        print("The opponent passed: {}".format(self.board[3][0][0]))
    

    def step(self, action):
        ''' return observation, reward, done, info '''
        pass
    
    def reset(self):
        pass
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
