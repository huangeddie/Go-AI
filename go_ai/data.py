import numpy as np
import gym
from go_ai import mcts
from go_ai.models import make_mcts_forward

go_env = gym.make('gym_go:go-v0', size=0)
govars = go_env.govars
gogame = go_env.gogame


def add_to_replay_mem(replay_mem, state, action_1d, next_state, reward, done, win, mcts_action_probs,
                      add_symmetries=True):
    """
    Adds original event, plus augmented versions of those events
    States are assumed to be (6, BOARD_SIZE, BOARD_SIZE)
    """
    assert len(state.shape) == 3 and state.shape[1] == state.shape[2]
    board_size = state.shape[1]
    num_channels = state.shape[0]

    if action_1d < board_size ** 2:
        action_2d_one_hot = np.eye(board_size ** 2)[action_1d].reshape(1, board_size, board_size)
    else:
        action_2d_one_hot = np.zeros((1, board_size, board_size))

    mc_board_pi = mcts_action_probs[:-1].reshape(1, board_size, board_size)

    chunk = np.concatenate([state, action_2d_one_hot, next_state, mc_board_pi], axis=0)

    orientations = gogame.get_symmetries(chunk) if add_symmetries else [chunk]
    for oriented_chunk in orientations:
        s = oriented_chunk[:num_channels]
        a = oriented_chunk[num_channels]
        if np.count_nonzero(a) == 0:
            a = board_size ** 2
        else:
            a = np.argmax(a)
        ns = oriented_chunk[num_channels + 1:(num_channels + 1) + num_channels]
        assert (num_channels + 1) + num_channels == oriented_chunk.shape[0] - 1
        mc_pi = oriented_chunk[-1].flatten()
        mc_pi = np.append(mc_pi, mcts_action_probs[-1])
        replay_mem.append((s, a, ns, reward, done, win, mc_pi))


def replay_mem_to_numpy(replay_mem):
    """
    Turns states from (BATCH_SIZE, 6, BOARD_SIZE, BOARD_SIZE) to (BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 6)
    :param replay_mem:
    :return:
    """
    replay_mem = list(zip(*replay_mem))
    states = np.array(list(replay_mem[0]), dtype=np.float32).transpose(0, 2, 3, 1)
    actions = np.array(list(replay_mem[1]), dtype=np.int)
    next_states = np.array(list(replay_mem[2]), dtype=np.float32).transpose(0, 2, 3, 1)
    rewards = np.array(list(replay_mem[3]), dtype=np.float32).reshape((-1,))
    terminals = np.array(list(replay_mem[4]), dtype=np.uint8)
    wins = np.array(list(replay_mem[5]), dtype=np.int)
    mct_pis = np.array(list(replay_mem[6]), dtype=np.float32)

    return states, actions, next_states, rewards, terminals, wins, mct_pis


def black_winning(info):
    if info['area']['b'] > info['area']['w']:
        black_won = 1
    elif info['area']['b'] < info['area']['w']:
        black_won = -1
    else:
        black_won = 0

    return black_won


def self_play(go_env, policy, max_steps, mc_sims, temp_func, get_symmetries=True):
    """
    Plays out a game, by pitting the policy against itself,
    and adds the events to the given replay memory
    :param go_env:
    :param policy:
    :param max_steps:
    :param mc_sims:
    :param get_symmetries:
    :return: The trajectory of events and number of steps
    """

    # Basic setup
    num_steps = 0
    state = go_env.reset()

    mem_cache = []

    mcts_forward = make_mcts_forward(policy)
    mct = mcts.MCTree(state, mcts_forward)

    while True:
        # Get turn
        curr_turn = go_env.turn

        # Get canonical state for policy and memory
        canonical_state = go_env.gogame.get_canonical_form(state, curr_turn)

        # Get action from MCT
        temp = temp_func(num_steps)
        mcts_action_probs = mct.get_action_probs(max_num_searches=mc_sims, temp=temp)
        action = gogame.random_weighted_action(mcts_action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, info = go_env.step(action)
        mct.step(action)

        # Get canonical form of next state for memory
        canonical_next_state = go_env.gogame.get_canonical_form(next_state, curr_turn)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        mem_cache.append((curr_turn, canonical_state, action, canonical_next_state, reward, done, mcts_action_probs))

        # Increment steps
        num_steps += 1

        # Max number of steps or game ended by consecutive passing
        if done:
            break

        # Setup for next event
        state = next_state

    assert done

    black_won = black_winning(info)

    # Add the last event to memory
    replay_mem = []
    for turn, canonical_state, action, canonical_next_state, reward, done, mcts_action_probs in mem_cache:
        if turn == go_env.govars.BLACK:
            win = black_won
        else:
            win = -black_won
        add_to_replay_mem(replay_mem, canonical_state, action, canonical_next_state, reward, done, win,
                          mcts_action_probs, add_symmetries=get_symmetries)

    # Game ended
    return replay_mem, num_steps


def pit(go_env, black_policy, white_policy, max_steps, mc_sims, temp_func):
    num_steps = 0
    state = go_env.reset()

    black_forward = make_mcts_forward(black_policy)
    white_forward = make_mcts_forward(white_policy)
    black_mct = mcts.MCTree(state, black_forward)
    white_mct = mcts.MCTree(state, white_forward)

    while True:
        # Get turn
        curr_turn = go_env.turn

        # Get an action
        temp = temp_func(num_steps)
        if curr_turn == go_env.govars.BLACK:
            mcts_action_probs = black_mct.get_action_probs(max_num_searches=mc_sims, temp=temp)
        else:
            assert curr_turn == go_env.govars.WHITE
            mcts_action_probs = white_mct.get_action_probs(max_num_searches=mc_sims, temp=temp)
        action = gogame.random_weighted_action(mcts_action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, info = go_env.step(action)

        # Sync the MC Trees
        black_mct.step(action)
        white_mct.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Increment steps
        num_steps += 1

        # Max number of steps or game ended by consecutive passing
        if done:
            break

    assert done

    black_won = black_winning(info)

    return black_won


def play_against(policy, go_env, mc_sims, temp):
    state = go_env.reset()
    mct_forward = make_mcts_forward(policy)
    mct = mcts.MCTree(state, mct_forward)

    while not go_env.game_ended:
        go_env.render()

        # Model's move
        mcts_action_probs = mct.get_action_probs(max_num_searches=mc_sims, temp=temp)
        action = gogame.random_weighted_action(mcts_action_probs)

        go_env.step(action)
        mct.step(action)

        go_env.render()

        # Player's move
        player_action = None
        valid_moves = go_env.get_valid_moves()
        while True:
            coords = input("Enter coordinates separated by space (`q` to quit)\n")
            if coords == 'q':
                return
            elif coords == 'p':
                player_action = None
            else:
                try:
                    coords = coords.split()
                    row = int(coords[0])
                    col = int(coords[1])
                    player_action = (row, col)
                except Exception as e:
                    print(e)
            player_action = go_env.action_2d_to_1d(player_action)
            if valid_moves[player_action]:
                break
            else:
                print("Invalid action")

        go_env.step(player_action)
        mct.step(player_action)
