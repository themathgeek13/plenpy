# Copyright (C) 2018-2020  The Plenpy Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Module defining image demosaicing methods.

This module is basically a wrapper of the ``color_demosaicing`` package,
see also the project's GitHub page:
https://github.com/colour-science/colour-demosaicing .

"""

import numpy as np
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007, )
from numpy.core.multiarray import ndarray
from scipy.ndimage.filters import convolve

from plenpy import spectral as hs


def get_demosaiced(img: ndarray,
                   pattern: str = 'GRBG',
                   method: str = 'bilinear') -> ndarray:
    """Get a demosaiced RGB image from a raw image.

    This function is a wrapper of the demosaicing functions supplied by the
    ``color_demosaicing`` package.

    Args:
        img: Input image, greyscale, of shape (x,y).

        pattern: Bayer filter pattern that the input image is modulated with.
            Patterns are: 'RGGB', 'BGGR', 'GRBG', 'GBRG'.

            Default: 'GRBG'

        method: Algorithm used to calculate the demosaiced image.\n
            * 'bilinear': Simple bilinear interpolation of color values
            * 'malvar2004': Algorithm introduced by Malvar et. al. [R3]_
            * 'menon2007': Algorithm introduced by Menon et. al. [R4]_,


    Returns:
        Demosaiced RGB-color image of shape (x,y,3) of
        dtype :class:`numpy.float64`.

    References:
        .. [R3]  H.S. Malvar,  Li-wei He, and  R. Cutler (2004).
           High-quality linear interpolation for demosaicing of
           Bayer-patterned color images.
           IEEE International Conference on Acoustics, Speech, and Signal
           Processing, Proceedings. (ICASSP '04).
           DOI: 10.1109/ICASSP.2004.1326587

        .. [R4]  D. Menon, S. Andriani, G. Calvagno (2007).
           Demosaicing With Directional Filtering and a posteriori Decision.
           IEEE Transactions on Image Processing (Volume: 16, Issue: 1)
           DOI: 10.1109/TIP.2006.884928

    """

    param_list = ["bilinear", "malvar2004", "menon2007"]

    # Do demosaicing with specified method
    if method not in param_list:
        raise ValueError(
            f"The specified method {method} is none of the supported "
            f"methods: {param_list}.")

    elif method == "bilinear":
        return demosaicing_CFA_Bayer_bilinear(img.astype(np.float64),
                                              pattern=pattern)

    elif method == "malvar2004":
        return demosaicing_CFA_Bayer_Malvar2004(img.astype(np.float64),
                                                pattern=pattern)

    elif method == "menon2007":
        return demosaicing_CFA_Bayer_Menon2007(img.astype(np.float64),
                                               pattern=pattern)


def hsi_demosaiced(img: ndarray,
                   pattern: ndarray,
                   method: str = 'bilinear') -> hs.SpectralImage:
    """Demosaicking method for hyperspectral images.

    Args:
        img: Input image,greyscale, of shape (x,y).

        pattern: Spectral filter array. Usually the size of the pattern needs
                 to be the same as the size of ``img``.
                 But for a uniform filter, a filter block is also feasible.

        method: Algorithm used to calculate the demosaiced image.

                * 'bilinear': Simple bilinear interpolation of color values separately.
                * 'bilinear_correlation': Bilinear interpolation of color values in one channel correlated with values in all channels.

    Returns:
        Demosaiced hyperspectral images of shape (x, y, channel) of
        as class:'HyperSpectralImage'

    """

    num_ch = np.amax(pattern) + 1
    if pattern.shape <= img.shape:
        pattern_all = np.tile(pattern,
                              (int(img.shape[0] // pattern.shape[0]) + 1, int(img.shape[1] // pattern.shape[1]) + 1))
        pattern = pattern_all[:img.shape[0], :img.shape[1]]
    else:
        raise ValueError(f"The sensor image or/und pattern is/are incorrect.")

    if method == 'bilinear_correlation':
        interp_image = bilinear_correlation(img, pattern, num_ch)
    elif method == 'bilinear':
        interp_image = bilinear(img, pattern, num_ch)
    else:
         raise ValueError(f"Unknown method {method}.")

    return hs.SpectralImage(interp_image)


def bilinear(sen_im, spec_filter, num_ch):
    """

    Args:
        sen_im: Sensor image.
        spec_filter: Spectral filter array.
        num_ch: Number of spectral channels of sensor image.

    Returns:
        Simple bilinear separately interpolated images.

    """
    interp_image = np.zeros((sen_im.shape[0], sen_im.shape[1], num_ch))
    for k in range(num_ch):
        chan_k = np.zeros((sen_im.shape[0], sen_im.shape[1]))
        for i in range(sen_im.shape[0]):
            for j in range(sen_im.shape[1]):
                if spec_filter[i, j] == k:
                    chan_k[i, j] = sen_im[i, j]
        interp_image[:, :, k] = convolve(chan_k, get_interpolation_kernel(num_ch))

    return interp_image


def bilinear_correlation(sen_im, spec_filter, num_ch):
    """

    Args:
        sen_im: Sensor image.
        spec_filter: Spectral filter array.
        num_ch: Number of spectral channels of sensor image.

    Returns:
        Bilinear interpolated images using values of all channels.

    """

    self_interp = bilinear(sen_im, spec_filter, num_ch)
    interpolated = np.zeros((self_interp.shape[0], self_interp.shape[1], self_interp.shape[2]))
    for b in range(self_interp.shape[2]):
        for a in range(self_interp.shape[2]):
            sen_im_odd = sen_im.copy()
            if b == a:
                interpolated[:, :, b] += mask_chan(sen_im_odd, spec_filter, b)
            else:
                chan_diff = mask_chan((self_interp[:, :, a] - sen_im), spec_filter, b)
                chan_diff_smo = convolve(chan_diff, get_interpolation_kernel(num_ch))
                interpolated[:, :, b] += mask_chan(sen_im - chan_diff_smo, spec_filter, a)
    return interpolated


def get_interpolation_kernel(num_ch):
    """

    Args:
        num_ch: Number of spectral channels of sensor image.

    Returns:
        Filter kernel

    """
    dim = np.sqrt(num_ch)
    if dim % 1 == 0:
        dim = int(dim)
        col = np.arange(1, 2 * dim).reshape(1, 2 * dim - 1)
        for i in range(dim, 2 * dim - 1):
            col[0, i] = col[0, 2 * (dim - 1) - i]
    else:
        raise ValueError(f"The number of channels is incorrect.")
    return (np.dot(col.T, col)) / (2 ** dim)


def mask_chan(image, spec_filter, chan):
    """

    Args:
        image: 2d Image.
        spec_filter: Spectral filter array.
        chan: Channel number.

    Returns:
        Updated image.
        Only this channel related values in image will be remained. Others will become zero.

    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if spec_filter[i, j] != chan:
                image[i, j] = 0
    return image
