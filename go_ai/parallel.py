import math
import sys
import time

from mpi4py import MPI
from tqdm import tqdm

from go_ai import game


def parallel_play(comm: MPI.Intracomm, go_env, pi1, pi2, gettraj, req_episodes):
    """
    Plays games in parallel
    :param comm:
    :param go_env:
    :param pi1:
    :param pi2:
    :param gettraj:
    :param req_episodes:
    :return:
    """
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    worker_episodes = int(math.ceil(req_episodes / world_size))
    episodes = worker_episodes * world_size
    single_worker = comm.Get_size() <= 1

    timestart = time.time()
    winrate, steps, traj = game.play_games(go_env, pi1, pi2, gettraj, worker_episodes, progress=single_worker)
    timeend = time.time()

    duration = timeend - timestart
    avg_time = comm.allreduce(duration / worker_episodes, op=MPI.SUM) / world_size
    winrate = comm.allreduce(winrate, op=MPI.SUM) / world_size
    avg_steps = comm.allreduce(sum(steps), op=MPI.SUM) / episodes

    parallel_err(rank, f'{pi1} V {pi2} | {episodes} GAMES, {avg_time:.1f} SEC/GAME, {avg_steps:.0f} STEPS/GAME, '
                       f'{100 * winrate:.1f}% WIN')
    return winrate, traj


def parallel_out(rank, s, rep=0):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    if rank == rep:
        print(s, flush=True)


def parallel_err(rank, s, rep=0):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    if rank == rep:
        tqdm.write(f"{time.strftime('%H:%M:%S', time.localtime())}\t{s}", file=sys.stderr)
        sys.stderr.flush()