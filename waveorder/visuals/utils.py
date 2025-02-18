import matplotlib.colors as mcolors
import numpy as np


# Main function to convert a complex-valued torch tensor to RGB numpy array
# with red at +1, green at +i, blue at -1, and purple at -i
def complex_tensor_to_rgb(array, saturate_clim_fraction=1.0):
    # Calculate magnitude and phase for the entire array
    magnitude = np.abs(array)
    phase = np.angle(array)

    # Normalize phase to [0, 1]
    hue = (phase + np.pi) / (2 * np.pi)
    hue = np.mod(hue + 0.5, 1)

    # Normalize magnitude to [0, 1] for saturation
    if saturate_clim_fraction is not None:
        max_abs_val = np.amax(magnitude) * saturate_clim_fraction
    else:
        max_abs_val = 1.0

    sat = magnitude / max_abs_val if max_abs_val != 0 else magnitude

    # Create HSV array: hue, saturation, value (value is set to 1)
    hsv = np.stack((hue, sat, np.ones_like(sat)), axis=-1)

    # Convert the entire HSV array to RGB using vectorized conversion
    rgb_array = mcolors.hsv_to_rgb(hsv)

    return rgb_array
