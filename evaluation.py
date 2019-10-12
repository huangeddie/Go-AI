import torch

from go_ai import policies, game


def evaluate(go_env, curr_pi: policies.Policy, checkpoint_pi: policies.Policy, num_games, checkpoint_path):
    # Evaluate against checkpoint model and other baselines
    opp_winrate, _ = game.play_games(go_env, curr_pi, checkpoint_pi, False, num_games)

    # Get the pytorch models
    curr_model = curr_pi.pytorch_model
    checkpoint_model = checkpoint_pi.pytorch_model
    assert isinstance(curr_model, torch.nn.Module)
    assert isinstance(checkpoint_model, torch.nn.Module)

    if opp_winrate > 0.6:
        # New parameters are significantly better. Accept it
        torch.save(curr_model.state_dict(), checkpoint_path)
        checkpoint_model.load_state_dict(torch.load(checkpoint_path))
        print(f"{100 * opp_winrate:.1f}% Accepted new model")
        return 1
    elif opp_winrate >= 0.4:
        # Keep trying
        print(f"{100 * opp_winrate:.1f}% Continuing to train current weights")
        return 0
    else:
        # New parameters are significantly worse. Reject it.
        curr_model.load_state_dict(torch.load(checkpoint_path))
        print(f"{100 * opp_winrate:.1f}% Rejected new model")
        return -1
