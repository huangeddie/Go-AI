import gym

from go_ai import policies, game, utils

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
base_model, base_pi = utils.create_model(args, 'Baseline', baseline=True)
print(f"Loaded model {base_pi}")

human_pi = policies.Human(args.render)

# Play
go_env.reset()
game.pit(go_env, base_pi, human_pi, False)
