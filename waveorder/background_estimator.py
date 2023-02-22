"""Estimate flat field images"""

import numpy as np
import itertools


"""

This script is adopted from 

https://github.com/mehta-lab/reconstruct-order


"""


class BackgroundEstimator2D:
    """Estimates flat field image"""

    def __init__(self, block_size=32):
        """
        Background images are estimated once per channel for 2D data
        :param int block_size: Size of blocks image will be divided into
        """

        if block_size is None:
            block_size = 32
        self.block_size = block_size

    def sample_block_medians(self, im):
        """Subdivide a 2D image in smaller blocks of size block_size and
        compute the median intensity value for each block. Any incomplete
        blocks (remainders of modulo operation) will be ignored.

        :param np.array im:         2D image
        :return np.array(float) sample_coords: Image coordinates for block
                                               centers
        :return np.array(float) sample_values: Median intensity values for
                                               blocks
        """

        im_shape = im.shape
        assert (
            self.block_size < im_shape[0]
        ), "Block size larger than image height"
        assert (
            self.block_size < im_shape[1]
        ), "Block size larger than image width"

        nbr_blocks_x = im_shape[0] // self.block_size
        nbr_blocks_y = im_shape[1] // self.block_size
        sample_coords = np.zeros(
            (nbr_blocks_x * nbr_blocks_y, 2), dtype=np.float64
        )
        sample_values = np.zeros(
            (nbr_blocks_x * nbr_blocks_y,), dtype=np.float64
        )
        for x in range(nbr_blocks_x):
            for y in range(nbr_blocks_y):
                idx = y * nbr_blocks_x + x
                sample_coords[idx, :] = [
                    x * self.block_size + (self.block_size - 1) / 2,
                    y * self.block_size + (self.block_size - 1) / 2,
                ]
                sample_values[idx] = np.median(
                    im[
                        x * self.block_size : (x + 1) * self.block_size,
                        y * self.block_size : (y + 1) * self.block_size,
                    ]
                )
        return sample_coords, sample_values

    @staticmethod
    def fit_polynomial_surface_2D(
        sample_coords, sample_values, im_shape, order=2, normalize=True
    ):
        """
        Given coordinates and corresponding values, this function will fit a
        2D polynomial of given order, then create a surface of given shape.

        :param np.array sample_coords: 2D sample coords (nbr of points, 2)
        :param np.array sample_values: Corresponding intensity values (nbr points,)
        :param tuple im_shape:         Shape of desired output surface (height, width)
        :param int order:              Order of polynomial (default 2)
        :param bool normalize:         Normalize surface by dividing by its mean
                                       for background correction (default True)

        :return np.array poly_surface: 2D surface of shape im_shape
        """
        assert (order + 1) * (order + 2) / 2 <= len(
            sample_values
        ), "Can't fit a higher degree polynomial than there are sampled values"
        # Number of coefficients is determined by (order + 1)*(order + 2)/2
        orders = np.arange(order + 1)
        variable_matrix = np.zeros(
            (sample_coords.shape[0], int((order + 1) * (order + 2) / 2))
        )
        order_pairs = list(itertools.product(orders, orders))
        # sum of orders of x,y <= order of the polynomial
        variable_iterator = itertools.filterfalse(
            lambda x: sum(x) > order, order_pairs
        )
        for idx, (m, n) in enumerate(variable_iterator):
            variable_matrix[:, idx] = (
                sample_coords[:, 0] ** n * sample_coords[:, 1] ** m
            )
        # Least squares fit of the points to the polynomial
        coeffs, _, _, _ = np.linalg.lstsq(
            variable_matrix, sample_values, rcond=-1
        )
        # Create a grid of image (x, y) coordinates
        x_mesh, y_mesh = np.meshgrid(
            np.linspace(0, im_shape[1] - 1, im_shape[1]),
            np.linspace(0, im_shape[0] - 1, im_shape[0]),
        )
        # Reconstruct the surface from the coefficients
        poly_surface = np.zeros(im_shape, np.float64)
        order_pairs = list(itertools.product(orders, orders))
        # sum of orders of x,y <= order of the polynomial
        variable_iterator = itertools.filterfalse(
            lambda x: sum(x) > order, order_pairs
        )
        for coeff, (m, n) in zip(coeffs, variable_iterator):
            poly_surface += coeff * x_mesh**m * y_mesh**n

        if normalize:
            poly_surface /= np.mean(poly_surface)
        return poly_surface

    def get_background(self, im, order=2, normalize=True):
        """
        Combine sampling and polynomial surface fit for background estimation.
        To background correct an image, divide it by background.

        :param np.array im:        2D image
        :param int order:          Order of polynomial (default 2)
        :param bool normalize:     Normalize surface by dividing by its mean
                                   for background correction (default True)

        :return np.array background:    Background image
        """

        coords, values = self.sample_block_medians(im=im)
        background = self.fit_polynomial_surface_2D(
            sample_coords=coords,
            sample_values=values,
            im_shape=im.shape,
            order=order,
            normalize=normalize,
        )
        # Backgrounds can't contain zeros or negative values
        # if background.min() <= 0:
        #     raise ValueError(
        #         "The generated background was not strictly positive {}.".format(
        #             background.min()),
        #     )
        return background


class BackgroundEstimator2D_GPU:
    """Estimates flat field image"""

    def __init__(self, block_size=32, gpu_id=0):
        """
        Background images are estimated once per channel for 2D data
        :param int block_size: Size of blocks image will be divided into
        """
        globals()["cp"] = __import__("cupy")
        self.gpu_id = gpu_id
        cp.cuda.Device(self.gpu_id).use()

        if block_size is None:
            block_size = 32
        self.block_size = block_size

    def sample_block_medians(self, im):
        """Subdivide a 2D image in smaller blocks of size block_size and
        compute the median intensity value for each block. Any incomplete
        blocks (remainders of modulo operation) will be ignored.

        :param np.array im:         2D image
        :return np.array(float) sample_coords: Image coordinates for block
                                               centers
        :return np.array(float) sample_values: Median intensity values for
                                               blocks
        """

        im_shape = im.shape
        assert (
            self.block_size < im_shape[0]
        ), "Block size larger than image height"
        assert (
            self.block_size < im_shape[1]
        ), "Block size larger than image width"

        nbr_blocks_x = im_shape[0] // self.block_size
        nbr_blocks_y = im_shape[1] // self.block_size
        sample_coords = np.zeros(
            (nbr_blocks_x * nbr_blocks_y, 2), dtype=cp.float64
        )
        sample_values = np.zeros(
            (nbr_blocks_x * nbr_blocks_y,), dtype=cp.float64
        )
        for x in range(nbr_blocks_x):
            for y in range(nbr_blocks_y):
                idx = y * nbr_blocks_x + x
                sample_coords[idx, :] = [
                    x * self.block_size + (self.block_size - 1) / 2,
                    y * self.block_size + (self.block_size - 1) / 2,
                ]
                sample_values[idx] = np.median(
                    im[
                        x * self.block_size : (x + 1) * self.block_size,
                        y * self.block_size : (y + 1) * self.block_size,
                    ]
                )
        return sample_coords, sample_values

    @staticmethod
    def median_cp(x):
        x = x.flatten()
        n = x.shape[0]
        s = cp.sort(x)
        m_odd = cp.take(s, n // 2)
        if n % 2 == 1:
            return m_odd
        else:
            m_even = cp.take(s, n // 2 - 1)
            return (m_odd + m_even) / 2

    @staticmethod
    def fit_polynomial_surface_2D(
        sample_coords, sample_values, im_shape, order=2, normalize=True
    ):
        """
        Given coordinates and corresponding values, this function will fit a
        2D polynomial of given order, then create a surface of given shape.

        :param np.array sample_coords: 2D sample coords (nbr of points, 2)
        :param np.array sample_values: Corresponding intensity values (nbr points,)
        :param tuple im_shape:         Shape of desired output surface (height, width)
        :param int order:              Order of polynomial (default 2)
        :param bool normalize:         Normalize surface by dividing by its mean
                                       for background correction (default True)

        :return np.array poly_surface: 2D surface of shape im_shape
        """
        assert (order + 1) * (order + 2) / 2 <= len(
            sample_values
        ), "Can't fit a higher degree polynomial than there are sampled values"
        # Number of coefficients is determined by (order + 1)*(order + 2)/2
        orders = np.arange(order + 1)
        variable_matrix = np.zeros(
            (sample_coords.shape[0], int((order + 1) * (order + 2) / 2))
        )
        order_pairs = list(itertools.product(orders, orders))
        # sum of orders of x,y <= order of the polynomial
        variable_iterator = itertools.filterfalse(
            lambda x: sum(x) > order, order_pairs
        )
        for idx, (m, n) in enumerate(variable_iterator):
            variable_matrix[:, idx] = (
                sample_coords[:, 0] ** n * sample_coords[:, 1] ** m
            )
        # Least squares fit of the points to the polynomial
        coeffs, _, _, _ = np.linalg.lstsq(
            variable_matrix, sample_values, rcond=-1
        )

        # Create a grid of image (x, y) coordinates
        x_mesh, y_mesh = cp.meshgrid(
            cp.linspace(0, im_shape[1] - 1, im_shape[1]),
            cp.linspace(0, im_shape[0] - 1, im_shape[0]),
        )
        # Reconstruct the surface from the coefficients
        poly_surface = cp.zeros(im_shape, cp.float)
        coeffs = cp.array(coeffs)
        order_pairs = list(itertools.product(orders, orders))
        # sum of orders of x,y <= order of the polynomial
        variable_iterator = itertools.filterfalse(
            lambda x: sum(x) > order, order_pairs
        )
        for coeff, (m, n) in zip(coeffs, variable_iterator):
            poly_surface += coeff * x_mesh**m * y_mesh**n

        if normalize:
            poly_surface /= cp.mean(poly_surface)
        return poly_surface

    def get_background(self, im, order=2, normalize=True):
        """
        Combine sampling and polynomial surface fit for background estimation.
        To background correct an image, divide it by background.

        :param np.array im:        2D image
        :param int order:          Order of polynomial (default 2)
        :param bool normalize:     Normalize surface by dividing by its mean
                                   for background correction (default True)

        :return np.array background:    Background image
        """
        cp.cuda.Device(self.gpu_id).use()
        im = cp.asnumpy(im)

        coords, values = self.sample_block_medians(im=im)
        background = self.fit_polynomial_surface_2D(
            sample_coords=coords,
            sample_values=values,
            im_shape=im.shape,
            order=order,
            normalize=normalize,
        )
        # Backgrounds can't contain zeros or negative values
        # if background.min() <= 0:
        #     raise ValueError(
        #         "The generated background was not strictly positive {}.".format(
        #             background.min()),
        #     )
        return background
