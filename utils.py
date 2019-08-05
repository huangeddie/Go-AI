import io
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

def get_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    Assumes shape to be [BATCH SIZE, BOARD SIZE, BOARD SIZE, 4]
    """
    board_size = states.shape[1]
    invalid_moves = states[:,:,:,2].reshape((-1, board_size**2))
    invalid_moves = np.insert(invalid_moves, board_size**2, 0, axis=1)
    return invalid_moves

def get_invalid_values(states):
    """
    Returns the action values of the states where invalid moves have -infinity value (minimum value of float32)
    and valid moves have 0 value
    """
    invalid_moves = get_invalid_moves(states)
    invalid_values = np.finfo(np.float32).min * invalid_moves
    return invalid_values

def horizontally_flip(state_or_action, board_size):
    if isinstance(state_or_action, np.ndarray):
        return np.flip(state_or_action, 2)
    else:
        if state_or_action == board_size**2:
            return state_or_action
        col = state_or_action % board_size
        flipped_action = state_or_action - col + (board_size-1 - col)
        return flipped_action
    
def vertically_flip(state_or_action, board_size):
    if isinstance(state_or_action, np.ndarray):
        return np.flip(state_or_action, 1)
    else:
        if state_or_action == board_size**2:
            return state_or_action
        row = state_or_action // board_size
        col = state_or_action % board_size
        flipped_action = (board_size-1 - row) * board_size + col
        return flipped_action
    
def rotate_90(state_or_action, board_size):
    if isinstance(state_or_action, np.ndarray):
        return np.rot90(state_or_action, axes=(1,2))
    else:
        row = state_or_action // board_size
        col = state_or_action % board_size
        rotated_action = (board_size-1 - col) * board_size + row
        return rotated_action
    
def all_orientations(state_or_action, board_size):
    orientations = []
    
    v_flip = vertically_flip(state_or_action, board_size)
    h_flip = horizontally_flip(state_or_action, board_size)
    
    
    rot_90 = rotate_90(state_or_action, board_size)
    rot_180 = rotate_90(rot_90, board_size)
    rot_270 = rotate_90(rot_180, board_size)
    
    x_flip = horizontally_flip(v_flip, board_size)
    d_flip = horizontally_flip(rot_90, board_size)
    m_flip = rotate_90(h_flip, board_size)
    
    # vertical, horizontal flip
    orientations.append(v_flip)
    orientations.append(h_flip)
    
    # Rotations
    orientations.append(rot_90)
    orientations.append(rot_270)
    
    # Diagonal and cross flip
    orientations.append(d_flip)
    orientations.append(x_flip)
    
    # Mirror and Identity
    orientations.append(m_flip)
    orientations.append(state_or_action)
    
    return orientations

def plot_move_distr(title, move_distr, scalar=None):
    """
    Takes in a 1d array of move values and plots its heatmap
    """
    board_size = int((len(move_distr) - 1) ** 0.5)
    plt.axis('off')
    plt.title('{} {:.1f}\n{:.1f}L {:.1f}H {:.1f}P'
              .format(title, 0 if scalar is None else scalar, 
                      np.min(move_distr[:-1]), np.max(move_distr[:-1]), 
                      move_distr[-1].numpy()))
    plt.imshow(np.reshape(move_distr[:-1], (board_size, board_size)))
    
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def random_weighted_action(move_weights):
    """
    Assumes all invalid moves have weight 0
    Action is 1D
    Expected shape is (1, NUM OF MOVES)
    """
    move_weights = preprocessing.normalize(move_weights, norm='l1')
    return np.random.choice(np.arange(len(move_weights[0])), p=move_weights[0])

def random_action(state):
    """
    Assumed to be (BOARD_SIZE, BOARD_SIZE, 4)
    Action is 1D
    """
    invalid_moves = state[:,:,2].flatten()
    invalid_moves = np.append(invalid_moves, 0)
    move_weights = 1 - invalid_moves

    return random_weighted_action(move_weights.reshape((1,-1)))
