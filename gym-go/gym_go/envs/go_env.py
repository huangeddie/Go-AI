import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from betago.dataloader.goboard import GoBoard

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
        # access: [Black, White, Ko, Passed][Row number][Column number]
        self.board_info = np.zeros((4, self.board_width, self.board_width), dtype=np.int)
        
        # use GoBoard from BetaGo for game status keeping 
        self.go_board = GoBoard(self.board_width)
        
        self.reward_method = reward_method

        # black goes first
        self.curr_player = 'b'
        

    def print_state(self):
        print("Your pieces (black):")
        print(self.board_info[0])
        print("Opponent's pieces (white):")
        print(self.board_info[1])
        print("Ko-protection:")
        print(self.board_info[2])
        print("The opponent passed: {}".format(self.board_info[3][0][0]))
    

    def step(self, action):
        ''' 
        Assumes the correct player is making a move. First is black.
        return observation, reward, done, info 
        '''
        
        # sanity check for the move
        if self.go_board.is_move_on_board(action):
            self.print_state()
            raise Exception('Move {} is illegal: there is already a piece at this location'.format(action))
        if self.go_board.is_move_suicide(self.curr_player, action):
            self.print_state()
            raise Exception('Move {} is illegal: this move is suicide'.format(action))
        if self.go_board.is_simple_ko(self.curr_player, action):
            self.print_state()
            raise Exception('Move {} is illegal: this location is a Ko'.format(action))

        # update whether previous player have passed
        self.board_info[3].fill(int(self.prev_player_passed))

        # check if the current player passes
        if action is None:
            self.update_board_info()
            self.prev_player_passed = True
            return self.board_info
        else:
            self.prev_player_passed = False

        # TODO add apply_move

    def update_board_info(self):
        '''
        Update Black, white, Ko
        '''

        # reset board info
        self.board_info.fill(0)

        # update board pieces
        for move, player in self.go_board.board.items():
            if player == 'b':
                self.board_info[0, move[0], move[1]] = 1
            elif player == 'w':
                self.board_info[1, move[0], move[1]] = 1

        # update Ko protection
        for r in range(self.board_width):
            for c in range(self.board_width):
                if self.go_board.is_simple_ko(self.curr_player, (r, c)):
                    self.board_info[2, r, c] = 1

    
    def reset(self):
        pass
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
