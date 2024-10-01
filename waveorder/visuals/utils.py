import numpy as np
import torch
import matplotlib.colors as mcolors


# Main function to convert a complex-valued torch tensor to RGB numpy array
def complex_tensor_to_rgb(tensor):
    # Convert the torch tensor to a numpy array
    tensor_np = tensor.numpy()
    
    # Calculate magnitude and phase for the entire array
    magnitude = np.abs(tensor_np)
    phase = np.angle(tensor_np)
    
    # Normalize phase to [0, 1] with red at 0
    hue = phase / (2 * np.pi) + 0.5
    
    # Normalize magnitude to [0, 1] for saturation
    max_abs_val = np.amax(magnitude)
    sat = magnitude / max_abs_val if max_abs_val != 0 else magnitude
    
    # Create HSV array: hue, saturation, value (value is set to 1)
    hsv = np.stack((hue, sat, np.ones_like(sat)), axis=-1)
    
    # Convert the entire HSV array to RGB using vectorized conversion
    rgb_array = mcolors.hsv_to_rgb(hsv)
    
    return rgb_array
