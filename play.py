import gym

from go_ai import policies, game, utils

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint', load_checkpoint=True)
print("Loaded model")

# Play
go_env.reset()
game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
