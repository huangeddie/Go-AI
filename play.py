import gym

from go_ai import policies, game

# Environment
BOARD_SIZE = 4
go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE)

# Policies
random_policy = policies.RandomPolicy()
greedy_policy = policies.QTempPolicy('Greedy', policies.greedy_val_func, temp=0)
greedy_mct_policy = policies.MctPolicy('MCT', 4, policies.greedy_val_func, temp=0, num_searches=128)
human_policy = policies.HumanPolicy()

game.pit(go_env, human_policy, greedy_mct_policy, False)
