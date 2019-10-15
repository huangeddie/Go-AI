import gym

from go_ai import policies, game
from go_ai.models import value_models
from hyperparameters import *

# Environment
BOARD_SIZE = 4
go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

# Policies
checkpoint_model = value_models.ValueNet(BOARD_SIZE)
checkpoint_model.load_state_dict(torch.load(CHECKPOINT_PATH))

# Policies
checkpoint_pi = policies.MctPolicy('Checkpoint', checkpoint_model, 8 * BOARD_SIZE ** 2, 0)

game.pit(go_env, policies.HUMAN_PI, checkpoint_pi, False)
