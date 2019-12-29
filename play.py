import gym
from mpi4py import MPI

from go_ai import policies, game, utils

args = utils.hyperparameters(MPI.COMM_WORLD)

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
modeldir = 'bin/checkpoints/2019-12-25/'
model, policy = utils.create_model(args, 'Checkpoint', modeldir=modeldir)
print(f"Loaded model {policy}")

human_pi = policies.Human(args.render)

# Play
go_env.reset()
game.pit(go_env, policy, human_pi)
