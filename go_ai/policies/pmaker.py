from go_ai.models import value_model, actor_critic
from go_ai import policies
import tensorflow as tf
import gym

GoGame = gym.make('gym_go:go-v0', size=0).gogame

def make_policy(policy_args: policies.PolicyArgs):
    """
    :param policy_args: A dictionary of policy arguments
    :return: A policy
    """

    if policy_args.mode == 'qtemp':
        model = tf.keras.models.load_model(policy_args.model_path)
        val_func = value_model.make_val_func(model)
        policy = policies.QTempPolicy(val_func, policy_args.temperature)

    elif policy_args.mode == 'monte_carlo':
        model = tf.keras.models.load_model(policy_args.model_path)
        forward_func = actor_critic.make_forward_func(model)
        init_state = GoGame.get_init_board(policy_args.board_size)
        policy = policies.MctPolicy(forward_func, init_state)

    elif policy_args.mode == 'random':
        policy = policies.RandomPolicy()
    elif policy_args.mode == 'greedy':
        policy = policies.QTempPolicy(value_model.greedy_val_func, temp=0)
    elif policy_args.mode == 'human':
        policy = policies.HumanPolicy()
    else:
        raise Exception("Unknown policy mode")

    return policy
