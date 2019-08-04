def action_1d_to_2d(action_1d, board_width):
    """
    Converts 1D action to 2D or None if it's a pass
    """
    if action_1d == board_width**2:
        action = None
    else:
        action = (action_1d // board_width, action_1d % board_width)
    return action

def action_2d_to_1d(action_2d, board_width):
    if action_2d is None:
        action_1d = board_width**2
    else:
        action_1d = action_2d[0] * board_width + action_2d[1]
    return action_1d
