import gym

import go_ai.policies.baselines
from go_ai import game, utils

args = utils.hyperparameters(['customdir=bin/baselines/'])

# Environment
go_env = gym.make('gym_go:go-v0', size=args.size)

# Policies
policy, model = go_ai.policies.baselines.create_policy(args, 'Checkpoint')
print(f"Loaded model {policy}")

human_pi = go_ai.policies.baselines.Human(args.render)

# Play
go_env.reset()
game.pit(go_env, policy, human_pi)
