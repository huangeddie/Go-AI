import os
import shutil

import torch

from go_ai import policies, game


def update_checkpoint(go_env, first_pi: policies.Policy, second_pi: policies.Policy, num_games, checkpoint_path):
    """
    Writes the PyTorch model parameters of the best policy to the checkpoint
    :param go_env:
    :param first_pi:
    :param second_pi:
    :param num_games:
    :param checkpoint_path:
    :return:
    * 1 = if first policy was better and its parameters were written to checkpoint
    * 0 = no policy was significantly better than the other, so nothing was written
    * -1 = the second policy was better and its parameters were written to checkpoint
    """
    # Evaluate against checkpoint model and other baselines
    first_winrate, _ = game.play_games(go_env, first_pi, second_pi, False, num_games)

    # Get the pytorch models
    first_model = first_pi.pytorch_model
    second_model = second_pi.pytorch_model
    assert isinstance(first_model, torch.nn.Module)
    assert isinstance(second_model, torch.nn.Module)

    if first_winrate > 0.6:
        # First policy was best
        torch.save(first_model.state_dict(), checkpoint_path)
        return 1
    elif first_winrate >= 0.4:
        return 0
    else:
        assert first_winrate < 0.4
        # Second policy was best
        torch.save(second_model.state_dict(), checkpoint_path)
        return -1


def set_disk_params(load_params, checkpoint_path, tmp_path, model):
    """
    Updates the checkpooint parameters based on the given arguments,
    and syncs the temporary parameters with checkpoint
    :param load_params:
    :param checkpoint_path:
    :param tmp_path:
    :param model:
    :return:
    """
    if load_params:
        assert os.path.exists(checkpoint_path)
        print("Starting from checkpoint")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print("Initialized checkpoint")
    shutil.copy(checkpoint_path, tmp_path)
