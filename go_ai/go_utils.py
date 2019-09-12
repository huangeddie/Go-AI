import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

govars = gym.make('gym_go:go-v0', size=0).govars

def get_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    Assumes shape to be [BATCH SIZE, BOARD SIZE, BOARD SIZE, 6]
    """
    batch_size = states.shape[0]
    board_size = states.shape[1]
    invalid_moves = states[:,:,:,govars.INVD_CHNL].reshape((batch_size, -1))
    invalid_moves = np.insert(invalid_moves, board_size**2, 0, axis=1)
    return invalid_moves

def get_valid_moves(states):
    return 1 - get_invalid_moves(states)

def get_invalid_values(states):
    """
    Returns the action values of the states where invalid moves have -infinity value (minimum value of float32)
    and valid moves have 0 value
    """
    invalid_moves = get_invalid_moves(states)
    invalid_values = np.finfo(np.float32).min * invalid_moves
    return invalid_values
    
def all_orientations(state_or_action):
    orientations = []
    
    v_flip = np.flip(state_or_action, 1)
    h_flip = np.flip(state_or_action, 2)
    
    rot_90 = np.rot90(state_or_action, axes=(1,2))
    rot_180 = np.rot90(rot_90, axes=(1,2))
    rot_270 = np.rot90(rot_180, axes=(1,2))
    
    x_flip = np.flip(v_flip, 2)
    d_flip = np.flip(rot_90, 2)
    m_flip = np.rot90(h_flip, axes=(1,2))
    
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

def plot_move_distr(title, move_distr, valid_moves, scalar=None):
    """
    Takes in a 1d array of move values and plots its heatmap
    """
    board_size = int((len(move_distr) - 1) ** 0.5)
    plt.axis('off')
    valid_values = np.extract(valid_moves[:-1] == 1, move_distr[:-1])
    plt.title(title + (' ' if scalar is None else ' {:.3f}S').format(scalar) 
              + '\n{:.3f}L {:.3f}H {:.3f}P'.format(np.min(valid_values) 
                                                   if len(valid_values) > 0 else 0, 
                                                   np.max(valid_values) 
                                                   if len(valid_values) > 0 else 0, 
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
