import glob
import os
import tifffile as tiff
import numpy as np


def extract_reconstruction_parameters(reconstructor, magnification):

    attr_dict = {'phase_dimension': reconstructor.phase_deconv,
                 'pad_z': reconstructor.pad_z,
                 'n_objective_media': reconstructor.n_media,
                 'bg_correction_option': reconstructor.bg_option,
                 'objective_NA': reconstructor.NA_obj * reconstructor.n_media,
                 'condenser_NA': reconstructor.NA_illu * reconstructor.n_media,
                 'magnification': magnification,
                 'swing': reconstructor.chi if reconstructor.N_channel == 4 else reconstructor.chi / 2 / np.pi,
                 'pixel_size': reconstructor.ps * magnification}

    return attr_dict


def load_bg(bg_path, height, width, ROI=None):
    """
    Parameters
    ----------
    bg_path         : (str) path to the folder containing background images
    height          : (int) height of image in pixels
    width           : (int) width of image in pixels
    ROI             : (tuple)  ROI of the background images to use, if None, full FOV will be used

    Returns
    -------
    bg_data   : (ndarray) Array of background data w/ dimensions (N_channel, Y, X)
    """

    bg_paths = glob.glob(os.path.join(bg_path, '*.tif'))
    bg_paths.sort()
    bg_data = np.zeros([len(bg_paths), height, width])

    for i in range(len(bg_paths)):
        img = tiff.imread(bg_paths[i])

        if ROI is not None and ROI != (0, 0, height, width):
            bg_data[i, :, :] = np.mean(img[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]])

        else:
            bg_data[i, :, :] = img

    return bg_data


def create_grid_from_coordinates(xy_coords, rows, columns):

    coords = dict()
    coords_list = []
    for idx, pos in enumerate(xy_coords):
        coords[idx] = pos
        coords_list.append(pos)

    # sort by X and then by Y
    coords_list.sort(key=lambda x: x[0])
    coords_list.sort(key=lambda x: x[1])

    # reshape XY coordinates into their proper 2D shape
    grid = np.reshape(coords_list, (rows, columns, 2))
    pos_index_grid = np.zeros((rows,columns), 'uint16')
    keys = list(coords.keys())
    vals = list(coords.values())

    for row in range(rows):
        for col in range(columns):

            # append position index (key) into a final grid by indexed into the coordinate map (values)
            pos_index_grid[row, col] = keys[vals.index(list(grid[row, col]))]

    return pos_index_grid


class MockEmitter:
    def emit(self):
        pass
