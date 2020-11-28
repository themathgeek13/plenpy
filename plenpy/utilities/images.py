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
Module defining useful image related methods.

"""
from typing import Optional, Tuple, Union, List

import numpy as np
import scipy.fftpack
from numpy.core.multiarray import ndarray
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage import transform as tf
from skimage.color import rgb2gray

import plenpy.logg
from plenpy.utilities import core
from plenpy.utilities import kernels

logger = plenpy.logg.get_logger()

# Load PyFFTW if available
try:
    import pyfftw
except ImportError:
    logger.info("plenpy.utilities.images: No PyFFTW package found. "
                "Only numpy and scipy FFT implementations are available.")


def normalize_in_place(img: ndarray, by="area"):
    """Normalize a ndarray in place. Only for float type arrays.

    Args:
        img: Input ndarray.

        by : Used normalization method. Available methods are:\n
                * 'area': normalize to sum one
                * 'peak': normalize to maximum value
                * 'l2': normalize by L2 norm of the array
    """
    param_list = ["area", "peak", "l2"]

    if img.dtype not in [np.float16, np.float32, np.float64]:
        raise ValueError("In place normalization only allowed for float arrays.")

    if by not in param_list:
        raise ValueError(
            "Specified argument type is not one of the recognized "
            f"methods: {param_list}")

    if by == "area":
        img /= img.sum()

    elif by == "peak":
        img /= img.max()

    elif by == "l2":
        img /= np.linalg.norm(img, ord='fro')


def normalize(img: ndarray, by="area", dtype=np.float64) -> ndarray:
    """Get a normalized copy of an ndarray, e.g. an image or kernel.
    The returned array will be a float.

    Args:
        img: Input ndarray.

        by : Used normalization method. Available methods are:\n
            * 'area': normalize to sum one
            * 'peak': normalize to maximum value
            * 'l2': normalize by L2 norm of the array

        dtype : Output dtype. Either np.float16, np.float32 or np.float64.
                If input is float, output will be of same word size.

    Returns:
        Normalized array of same shape as input array.

    See Also:
        Based on :func:`normalize_in_place()`.

    """
    if img.dtype not in [np.float16, np.float32, np.float64]:
        d_type = dtype
    else:
        d_type = img.dtype

    res = img.copy().astype(d_type)
    normalize_in_place(res, by=by)

    return res


def fourier(img: ndarray,
            shift: bool = True,
            window: Optional[str] = None,
            pad: Optional[Union[int, List[int], ndarray]] = None,
            implementation: str = 'scipy') -> ndarray:
    """Calculates the fourier transform of an image.
    Image will be downsampled to greyscale if necessary.

    Additionally to the basic fast fourier transformations (FFT), such as
    numpy or scipy, this function implements proper windowing prior to
    the transformation, as well as a proper shift of the zero frequency
    component for even-sized images. Different implementations are supported.
    For plotting, see `func:plenpy.utilities.plots.plot_fft`.

    Note, that color channels will be converted to greyscale prior to FFT
    calculation.

    Args:
        img: Input image.

        shift: If ``True``, shift spectrum, i.e. zero frequency
            component is shifted to the center. If image is even-sized, zero
            padding will be performed.

        window: Specify 2D window used to reduce spectral leakage.
            See :func:`.plenpy.utilities.kernels.get_avail_names()` for a list
            of available windows kernels. If ``None`` is specified,
            no windowing is performed.

        pad: Zero padding applied in all directions in pixel.
            If int is given, padding will be identical in all directions.
            If list/ndarray ist given, padding will be performed independently
            in x- and y-direction.

        implementation: The implementation used to calculate the Fourier
            transform. Available implementations are 'scipy', 'numpy', 'fftw'.
            See also: :func:`.scipy.fftpack.fft2()` and
            :func:`.numpy.fft.fft2()`. For 'fftw', ``pyFFTW`` has to be
            installed on the system, see https://github.com/hgomersall/pyFFTW .
            Default: 'scipy'.

    Returns:
        The (complex valued) fourier transform of the input image.

    """

    implementation_list = ['numpy', 'scipy', 'fftw']

    # Check image shape, convert to greyscale if necessary
    if img.ndim == 3:
        img = rgb2gray(img)
    n, m = img.shape
    
    if window is not None:
        # Get specified 2D window function
        kern = kernels.get_kernel(window, size=n, size_y=m, normalize='peak').astype(img.dtype)

        # Do element wise window multiplication
        img = np.multiply(kern, img)

    # If shape is even, add zero padding to center the zero frequency
    if n % 2 == 0:
        img = np.pad(img, ((0, 1), (0, 0)), mode='constant')
        n, m = img.shape

    if m % 2 == 0:
        img = np.pad(img, ((0, 0), (0, 1)), mode='constant')
        n, m = img.shape

    if pad is not None:
        # Perform zero padding
        if type(pad) == int:
            pad_x = pad
            pad_y = pad
        elif (type(pad) == list or type(pad) == ndarray) and len(pad) == 2:
            pad_x = pad[0]
            pad_y = pad[1]
        else:
            raise ValueError("Option 'pad' must be int or list/ndarray of length 2")

        # Perform padding
        img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
        n, m = img.shape

    if implementation not in implementation_list:
        raise ValueError(
            f"The implementation '{implementation}' is not one of the "
            f"supported implementations, {implementation_list}.")

    elif implementation == 'scipy':
        img = scipy.fftpack.fft2(img)

    elif implementation == 'numpy':
        img = np.fft.fft2(img)

    elif implementation == 'fftw':
        pyfftw.interfaces.cache.enable()
        img = pyfftw.interfaces.numpy_fft.fft2(img,
                                               threads=4,
                                               planner_effort='FFTW_ESTIMATE')

    if shift:
        img = np.fft.fftshift(img)

    return img


def crop_center(img: ndarray,
                crop_x: int,
                crop_y: Optional[int] = None) -> ndarray:
    """Utility function to crop an image around its center.

    Args:
        img: Input image.

        crop_x: Size of the cropped image in x-direction

        crop_y: Size of the cropped image in y-direction.
            If ``None``, a quadratic crop is returned.

    Returns:
        Cropped Image of shape (cropped_x, cropped_y, num_channels).

    Notes:
        No interpolation is done at any point. That means in particular: if you
        pass an image of even dimensions and crop to odd dimensions, the
        returned image is not the exact center (this would need interpolation)
        but rather an image one pixel off the original image's center.
        Analogously, if you crop an image of odd dimension to an image of even
        dimensions, the same holds. The returned image will only be the true
        center of the original image if the crop sizes in both direction
        matches the original size dimensions in the modulo 2 sense
        (i.e. even/odd).
    """

    if crop_y is None:
        crop_y = crop_x

    # Bring image in shape (x, y, num_channels)
    if img.ndim == 2:
        x, y = img.shape
    else:
        x, y, dummy = img.shape

    # If crop_x and crop_y are larger than shape of image, throw error
    if x < crop_x or y < crop_y:
        raise ValueError(
            f"Crop dimension ({crop_x}{crop_y})larger than "
            f"input image shape {img.shape}.")

    # Crop image and return
    start_x = (x - crop_x) // 2
    start_y = (y - crop_y) // 2
    return img[start_x:start_x + crop_x, start_y:start_y + crop_y]


def shear(im: ndarray, k: float, order: int = 1):
    """Shear a 2D image. If k=1, vertical lines are at 45° after shearing.

    Args:
        im: 2D image to shear.
        k: Shearing factor.
        order: Interpolation order. Default: 1 (bilinear)

    Returns:
        Sheared image. Output shape is same as input.

    """
    x, y = im.shape[0], im.shape[1]

    # Affine shearing transform
    M = np.asarray([[1, -k, 0], [0, 1, 0], [0, 0, 1]])

    t_shear = tf.AffineTransform(matrix=M)
    t_trans = tf.AffineTransform(translation=(k*x/2, 0))
    t_tot = t_trans + t_shear

    return tf.warp(im, t_tot, order=order)


def overlay_images(img1: ndarray,
                   img2: ndarray) -> ndarray:
    """Overlay two images in grayscale mode for comparison.

    The used color scheme is magenta-green by default.
    That is, it shows ``img1`` in green and ``img2`` in magenta.
    The overlay is grey in areas where the two images have similar intensity.
    The images have to be of same shape. Color images will be converted to
    grayscale.

    Args:
        img1: Input image.

        img2: Input image.

    Returns:
        Overlay of ``img1`` and ``img2`` of shape (x, y, 3).

    Raises:
        ValueError:  Images ``img1`` and ``img2`` must be of same shape.
            Found shapes img1.shape and img2.shape .

    """
    if np.array_equal(img1.shape, img2.shape) is not True:
        raise ValueError(
            "Images img1 and img2 must be of same shape. "
            f"Found shapes {img1.shape} and {img2.shape}.")

    # Make images Grayscale
    image1 = rgb2gray(img1)
    image2 = rgb2gray(img2)

    # Make images color again, image1 in red, image2 in green
    x, y = image1.shape
    color_image1 = np.zeros((x, y, 3))
    color_image1[:, :, 0] = image1
    color_image1[:, :, 2] = image1

    x, y = image2.shape
    color_image2 = np.zeros((x, y, 3))
    color_image2[:, :, 1] = image2
    color_image2[:, :, 2] = image2

    # Merge images,
    # R channel -> image1, G channel -> image 2, B channel ->
    merge = np.zeros((x, y, 3))
    merge[:, :, 0] = color_image1[:, :, 0]
    merge[:, :, 1] = color_image2[:, :, 1]
    merge[:, :, 2] = np.clip(
        0.5*(color_image1[:, :, 2] + color_image2[:, :, 2]),
        a_min=0.0,
        a_max=1.0)

    return merge


def get_gradients(im: ndarray, method='sobel', **kwargs) -> Tuple[ndarray, ndarray]:
    """Get the gradients of a 2D-image using different methods.

    Args:
        im: Input image of shape (x, y). Only monochromatic images are supported here.
        method: Used method for gradient calculation. Possible values are:
                    * 'scharr': Scharr filter.
                    * 'sobel': Sobel filter.
                    * 'dog': Difference of Gaussians.
                    * 'gradient': Numpy's ``gradient()`` method

    Returns:
        Tuple (gx, gy).
        Gradient gx, gy in x- and y-direction, respectively.

    """
    im = im.squeeze()

    if im.ndim != 2:
        raise ValueError(
            "Gradient calculation only works on 2D images right now. "
            "For multi-dimensional arrays, use numpy's gradient() instead.")

    x, y = im.shape

    # Get image gradients
    param_list = ["scharr", "sobel", "dog", "gradient"]
    method = method.lower()

    if method not in param_list:
        raise ValueError(
            f"Specified method '{method}' is not one of the recognized "
            f"methods: {param_list}")

    if method == 'scharr':
        h = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        grad_x = convolve(im, h.T)
        grad_y = convolve(im, h)

    elif method == 'sobel':
        h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_x = convolve(im, h.T)
        grad_y = convolve(im, h)

    elif method == 'dog':
        if not 'sigma' in kwargs:
            sigma_1 = min(x, y)/800
        else:
            sigma_1 = kwargs['sigma']

        sigma_2 = 1.6 * sigma_1
        grad_x = gaussian_filter(im, [sigma_1, 0]) - gaussian_filter(im, [sigma_2, 0])
        grad_y = gaussian_filter(im, [0, sigma_1]) - gaussian_filter(im, [0, sigma_2])

    elif method == 'gradient':
        grad_y, grad_x = np.gradient(im)  # central differences

    return grad_x, grad_y


def get_edges(im: ndarray, method='sobel', **kwargs) -> ndarray:
    """Get the gradients of a 2D-image using different methods.

    Args:
        im: Input image of shape (x, y). Only monochromatic images are supported here.
        method: Used method for gradient calculation. Possible values are:
                    * 'scharr': Scharr filter.
                    * 'sobel': Sobel filter.
                    * 'dog': Difference of Gaussians.
                    * 'gradient': Numpy's ``gradient()`` method

    Returns:
        Tuple (gx, gy).
        Gradient gx, gy in x- and y-direction, respectively.

    """

    if im.ndim != 2:
        raise ValueError(
            "Gradient calculation only works on 2D images right now. "
            "For multi-dimensional arrays, use numpy's gradient() instead.")

    x, y = im.shape

    # Get image gradients
    param_list = ["scharr", "sobel", "gradient", "dog", "log"]
    method = method.lower()

    if method not in param_list:
        raise ValueError(
            f"Specified method '{method}' is not one of the recognized "
            f"methods: {param_list}")

    if method in ['scharr', 'sobel', 'gradient']:
        gx, gy = get_gradients(im, method=method)
        res = np.hypot(gx, gy)

    elif method == 'dog':
        if not 'sigma' in kwargs:
            sigma_1 = min(x, y)/800
        else:
            sigma_1 = kwargs['sigma']

        sigma_2 = 1.6 * sigma_1
        res = gaussian_filter(im, [sigma_1, sigma_1]) - gaussian_filter(im, [sigma_2, sigma_2])

    elif method == 'log':
        if not 'sigma' in kwargs:
            sigma = min(x, y)/800
        else:
            sigma = kwargs['sigma']

        # Convolve with Gaussian
        res = gaussian_filter(im, [sigma, sigma])

        # Apply Laplace
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        res = convolve(res, h)

    return res
