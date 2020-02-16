import multiprocessing as mp

import gym
import numpy as np
from tqdm import tqdm

from go_ai import utils
from go_ai.policies import baselines

size = 9
arg_strs = [
    f'--size={size} --model=val --temp=1 --mcts=0 --baseline',
    f'--size={size} --model=ac --temp=1 --mcts=-1 --baseline',
]


def evaluate_greedy_actions(all_args, replay, workers=8):
    context = mp.get_context('spawn')
    queue = context.Queue()
    processes = []
    n = len(replay)
    chunk = n // workers
    for rank in range(workers):
        worker_replay = replay[rank * chunk:(rank + 1) * chunk] if rank < workers - 1 else replay[rank * chunk:]
        p = context.Process(target=worker_eval_greedy, args=(queue, all_args, worker_replay))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_greedy_actions = []
    while not queue.empty():
        greedy_actions = queue.get()
        all_greedy_actions.append(greedy_actions)

    all_greedy_actions = np.concatenate(all_greedy_actions)

    return all_greedy_actions


def worker_eval_greedy(queue, all_args, replay):
    policies = []
    args_rep = all_args[0]
    for args in all_args:
        policy, _ = baselines.create_policy(args)
        policies.append(policy)

    go_env = gym.make('gym_go:go-v0', size=args_rep.size)
    all_greedy_actions = []
    for traj in tqdm(replay, desc='Greedy Actions'):
        go_env.reset()
        for a in traj.actions:
            greedy_actions = []
            for policy in policies:
                pi = policy(go_env)
                greedy_actions.append(np.argmax(pi))
            all_greedy_actions.append(greedy_actions)

            go_env.step(a)

    queue.put(np.array(all_greedy_actions, dtype=int))


if __name__ == '__main__':
    all_args = []

    for arg_str in arg_strs:
        args = utils.hyperparameters(arg_str.split())
        all_args.append(args)

    _, _, replay = utils.multi_proc_play(all_args[0], all_args[0], 16)

    greedy_actions = evaluate_greedy_actions(all_args, replay)

    best_actions = greedy_actions[:, -1]
    stats_str = ''
    for i in range(len(all_args) - 1):
        agreement = np.mean(greedy_actions[:, i] == best_actions)
        stats_str += (f'{100 * agreement:.1f}% ')
    print(stats_str)
