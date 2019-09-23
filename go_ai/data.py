import gym
from tqdm import tqdm
import numpy as np
import os
import shutil
import multiprocessing as mp
from go_ai import policies, models
import queue

go_env = gym.make('gym_go:go-v0', size=0)
govars = go_env.govars
gogame = go_env.gogame


def add_to_replay_mem(replay_mem, state, action_1d, next_state, reward, done, win, qvals, add_symmetries=True):
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

    pass_val = qvals[-1:]
    qvals = qvals[:-1].reshape(1, board_size, board_size)
    chunk = np.concatenate([state, action_2d_one_hot, next_state, qvals], axis=0)

    orientations = gogame.get_symmetries(chunk) if add_symmetries else [chunk]
    for oriented_chunk in orientations:
        s = oriented_chunk[:num_channels]
        a = oriented_chunk[num_channels]
        if np.count_nonzero(a) == 0:
            a = board_size ** 2
        else:
            a = np.argmax(a)
        ns = oriented_chunk[num_channels + 1: 2 * num_channels + 1]
        assert ns.shape[0] == num_channels
        assert 2*num_channels+1 == oriented_chunk.shape[0]-1
        qvals = oriented_chunk[-1]
        qvals = np.concatenate([qvals.flatten(), pass_val])
        replay_mem.append((s, a, ns, reward, done, win, qvals))

def add_traj_to_replay_mem(replay_mem, black_won, trajectory, val_func, add_symmetries):
    for turn, canonical_state, action, canonical_next_state, reward, done in trajectory:
        if turn == go_env.govars.BLACK:
            win = black_won
        else:
            win = -black_won
        qvals = models.get_qvals(canonical_state[np.newaxis], val_func)[0]
        add_to_replay_mem(replay_mem, canonical_state, action, canonical_next_state, reward, done, win, qvals,
                          add_symmetries)


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
    qvals = np.array(list(replay_mem[6]), dtype=np.float32)

    return states, actions, next_states, rewards, terminals, wins, qvals


def pit(go_env, black_policy, white_policy, get_memory=False):
    """
    Pits two policies against each other and returns the results
    :param go_env:
    :param black_policy:
    :param white_policy:
    :param mc_sims:
    :param temp_func:
    :return: Whether or not black won {1, 0, -1}, trajectory
        trajectory is a list of events where each event is of the form
        (curr_turn, canonical_state, action, canonical_next_state, reward, done)
    """
    num_steps = 0
    state = go_env.reset()
    black_policy.reset()
    white_policy.reset()

    max_steps = 2 * (go_env.size ** 2)

    mem_cache = []

    done = False
    info = None
    while not done:
        # Get turn
        curr_turn = go_env.turn

        # Get canonical state for policy and memory
        canonical_state = go_env.gogame.get_canonical_form(state, curr_turn)

        # Get an action
        if curr_turn == go_env.govars.BLACK:
            action_probs = black_policy(state, step=num_steps)
        else:
            assert curr_turn == go_env.govars.WHITE
            action_probs = white_policy(state, step=num_steps)
        action = gogame.random_weighted_action(action_probs)

        # Execute actions in environment and MCT tree
        next_state, reward, done, info = go_env.step(action)

        # Get canonical form of next state for memory
        canonical_next_state = go_env.gogame.get_canonical_form(next_state, curr_turn)

        # Sync the policies
        black_policy.step(action)
        if white_policy != black_policy:
            white_policy.step(action)

        # End if we've reached max steps
        if num_steps >= max_steps:
            done = True

        # Add to memory cache
        if get_memory:
            mem_cache.append((curr_turn, canonical_state, action, canonical_next_state, reward, done))

        # Increment steps
        num_steps += 1

        # Setup for next event
        state = next_state

    assert done

    # Determine who won
    if info['area']['b'] > info['area']['w']:
        black_won = 1
    elif info['area']['b'] < info['area']['w']:
        black_won = -1
    else:
        black_won = 0

    return black_won, mem_cache


def self_play(go_env, policy, get_memory=False):
    """
    Plays out a game, by pitting the policy against itself,
    and adds the events to the given replay memory
    :param go_env:
    :param policy:
    :param mc_sims:
    :param get_symmetries:
    :return: The trajectory of events and number of steps
    """
    return pit(go_env, black_policy=policy, white_policy=policy, get_memory=get_memory)


def eps_job(episode_queue, first_policy_won_queue, board_size, first_policy_args, second_policy_args, val_func_args,
            out):
    """
    Continously executes episode jobs from the episode job queue until there are no more jobs
    :param episode_queue:
    :param first_policy_won_queue:
    :param board_size:
    :param first_policy_args:
    :param second_policy_args:
    :param out: If outpath is specified, the replay memory is store in that path
    :return:
    """
    go_env = gym.make('gym_go:go-v0', size=board_size)
    first_policy = policies.make_policy(first_policy_args, board_size)
    if first_policy_args == second_policy_args:
        second_policy = first_policy
    else:
        second_policy = policies.make_policy(second_policy_args, board_size)

    actor_critic = models.make_actor_critic(go_env.size)
    actor_critic.load_weights(val_func_args['weights_path'])
    val_func = models.make_val_func(actor_critic)

    replay_mem = []
    while not episode_queue.empty():
        try:
            first_policy_black = episode_queue.get()
            if first_policy_black:
                black_policy, white_policy = first_policy, second_policy
            else:
                black_policy, white_policy = second_policy, first_policy
            black_won, trajectory = pit(go_env, black_policy=black_policy, white_policy=white_policy,
                                                   get_memory=out is not None)

            # Add trajectory to replay memory
            add_traj_to_replay_mem(replay_mem, black_won, trajectory, val_func, add_symmetries=True)

            first_policy_won = black_won if first_policy_black else -black_won
            first_policy_won_queue.put((first_policy_won + 1) / 2)
        except Exception as e:
            print(e)

    if out is not None and len(replay_mem) > 0:
        np.random.shuffle(replay_mem)
        data = replay_mem_to_numpy(replay_mem)
        np.savez(out, *data)


def make_episodes(board_size, first_policy_args, second_policy_args, val_func_args, episodes, num_workers, outdir=None):
    """
    Multiprocessing of pitting the first policy against the second policy
    :param board_size:
    :param first_policy_args:
    :param second_policy_args:
    :param episodes:
    :param num_workers:
    :param outdir:
    :return: The win rate of the first policy against the second policy
    """
    ctx = mp.get_context('spawn')

    # Create job queue specific to whether or not we're doing multiprocessing
    if num_workers > 1:
        queue_maker = ctx
    else:
        queue_maker = queue

    episode_queue = queue_maker.Queue(maxsize=episodes)
    first_policy_won_queue = queue_maker.Queue(maxsize=episodes)

    # Set the episode jobs
    for i in range(episodes):
        # Puts whether the first or second policy should be black
        episode_queue.put(i % 2 == 0)

    # Set the parameters for the workers
    if outdir is not None:
        # Remove old data if it exists
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
            os.makedirs(outdir)
        worker_out = os.path.join(outdir, 'worker_0.npz')

    else:
        worker_out = None
    default_eps_job_args = (episode_queue, first_policy_won_queue, board_size, first_policy_args, second_policy_args,
                            val_func_args, worker_out)

    # Launch the workers
    processes = []
    if num_workers > 1:
        for i in range(num_workers):
            # Set the parameters for the workers
            if outdir is not None:
                worker_out = os.path.join(outdir, 'worker_{}.npz'.format(i))
            else:
                worker_out = None
            eps_job_args = (episode_queue, first_policy_won_queue, board_size, first_policy_args, second_policy_args,
                            val_func_args, worker_out)

            p = ctx.Process(target=eps_job, args=eps_job_args)
            p.start()
            processes.append(p)
    else:
        eps_job(*default_eps_job_args)

    # Collect the results
    wins = []
    pbar = tqdm(range(episodes), desc='Episodes')
    for _ in pbar:
        first_policy_won = first_policy_won_queue.get()
        wins.append(first_policy_won)
        pbar.set_postfix_str('{:.1f}%'.format(100 * np.mean(wins)))

    # Cleanup the workers if necessary
    for p in processes:
        p.join()

    return np.mean(wins)


def episodes_from_dir(episodes_dir):
    """
    episode_dir ->
        * worker_0.npz
        * worker_1.npz
        * ...

    worker_n.npz are files arrays of [states, actions, next_states, rewards, terminals, wins, mct_pis]
    :param episodes_dir:
    :return: all of [states, actions, next_states, rewards, terminals, wins, mct_pis] concatenated and shuffled
    """
    data = []
    for file in os.listdir(episodes_dir):
        if file[0] == '.':
            continue
        data_batch = np.load(os.path.join(episodes_dir, file))
        data.append([data_batch[file] for file in data_batch.files])
    foo = list(zip(*data))
    np_data = [np.concatenate(bar) for bar in foo]
    return np_data
