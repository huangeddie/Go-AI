import gym
import torch

import utils
from go_ai import policies, game
from go_ai.models import value, actorcritic

args = utils.hyperparameters()

# Environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize)

# Policies
checkpoint_model, checkpoint_pi = utils.create_agent(args, 'Checkpoint')
checkpoint_model.load_state_dict(torch.load(args.checkpath))
print("Loaded model")

# Play
go_env.reset()
game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
