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
        @param reward_method: either 'heuristic' or 'real' 
        heuristic: gives # black pieces - # white pieces. 
        real: gives 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, all from black player's perspective
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

        self.prev_player_passed = False

        # whether the game is done
        self.done = False
        

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
        
        # update whether previous player have passed
        self.board_info[3].fill(int(self.prev_player_passed))

        # if the current player passes
        if action is None:
            # if two consecutive passes, game is over
            if self.prev_player_passed:
                self.done = True

        # the current player makes a move
        else:
            # make sure the move is legal
            self.check_legal_move(action)

            # make the move and extract board info
            self.go_board.apply_move(self.curr_player, action)
            self.update_board_info()

            # switch player for the next round
            self.curr_player = self.go_board.other_color(self.curr_player)

        # store whether this player passed
        self.prev_player_passed = action is None

        return self.board_info, self.get_reward(), self.done, None


    def check_legal_move(self, action):
        '''
        Check: piece already on board, move is suicide, move is on ko
            move is out of board.
        '''
        exception_prefix = 'Move {} is illegal: '.format(action)

        # sanity check for the move
        if self.go_board.is_move_on_board(action):
            self.print_state()
            raise Exception(exception_prefix + 'there is already a piece at this location')
        if self.go_board.is_move_suicide(self.curr_player, action):
            self.print_state()
            raise Exception(exception_prefix + 'this move is suicide')
        if self.go_board.is_simple_ko(self.curr_player, action):
            self.print_state()
            raise Exception(exception_prefix + 'this location is a Ko')
        if not self.is_move_within_bounds(action):
            raise Exception(exception_prefix + 'out of board')


    def is_move_within_bounds(self, action):
        return action[0] >= 0 and action[0] < self.board_width \
            and action[1] >= 0 and action[1] < self.board_width


    def get_reward(self):
        '''
        Return reward based on reward_method.
        heuristic: black total area - white total area
        real: 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, from black player's perspective.
            Winning and losing based on the Area rule
        Win definition: https://en.wikipedia.org/wiki/Rules_of_Go#End
        '''
        if self.reward_method == 'real':
            if self.done:
                final_reward = self.get_area_reward() 

                if final_reward == 0:
                    return 0
                elif final_reward > 0:
                    return 1
                else:
                    return -1
            else: 
                return 0

        elif self.reward_method == 'heuristic':
            return self.get_area_reward()

        else:
            raise Exception('Unsupported reward method: {}'.format(self.reward_method))

    def get_area_reward(self):
        pass
        

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
