from go_ai.models import value_model, actor_critic
from go_ai.policies import GreedyPolicy, MctPolicy, RandomPolicy, HumanPolicy, PolicyArgs

def make_policy(policy_args: PolicyArgs):
    """
    :param policy_args: A dictionary of policy arguments
    :return: A policy
    """
    board_size = policy_args.board_size

    if policy_args.mode == 'values':
        model = value_model.make_val_net(board_size)
        model.load_weights(policy_args.weight_path)
        val_func = value_model.make_val_func(model)
        policy = GreedyPolicy(val_func)

    elif policy_args.mode == 'monte_carlo':
        model = actor_critic.make_actor_critic(board_size)
        model.load_weights(policy_args.weight_path)
        # TODO: Declare properly
        policy = MctPolicy(actor_critic, None, 0)

    elif policy_args.mode == 'random':
        policy = RandomPolicy()

    elif policy_args.mode == 'greedy':
        policy = GreedyPolicy(value_model.greedy_val_func)
    elif policy_args.mode == 'human':
        policy = HumanPolicy()
    else:
        raise Exception("Unknown policy mode")

    return policy
