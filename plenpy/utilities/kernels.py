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
Module defining different 2D kernels.

Feel free to add more kernels. To get list of all available kernels use
:func:`get_avail_names()`.

"""

from typing import List, Optional, Union

import numpy as np
from numpy.core.multiarray import ndarray
from scipy.special import iv

import plenpy.logg
from plenpy.utilities import images

logger = plenpy.logg.get_logger()

# Define a list of all allowed sensor image suffixes
# This corresponds to extensions that can be read by :func:`imageio.imread()`.
__availableKernelNames = ['disk', 'gauss', 'hann', 'hamming', 'kaiser',
                          'kaiser_rotational', 'hann_rotational']

__availableKernelNames = sorted(__availableKernelNames)


def get_avail_names() -> List[str]:
    """List all available kernel names.

    Returns:
        A list of all available kernel names that can for example
        be passed to :func:`get_kernel()`.

    """
    return __availableKernelNames


def get_kernel_disk(size: int,
                    radius: Optional[float] = None,
                    size_y: Optional[int] = None,
                    radius_y: Optional[int] = None,
                    normalize: str = 'area') -> ndarray:
    """Get a normalized 2D disk kernel.

    Args:
        size: Size of the kernel in pixels.

        radius: Radius of the disk.

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        radius_y: Radius of the disk in y-direction.
            If no value is specified, a circular disc is calculated.

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A disk kernel of shape (size, size_y) with radii
        (radius, radius_y).

    """
    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size
    radius_x = radius

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(f"Calculating disk kernel with shape ({size_x}, {size_y}).")

    # Set maximal radius if no values are specified
    if radius_x is None and radius_y is None:
        radius_x = min((size - 1)/2.0, (size_y - 1)/2.0)
        radius_y = radius_x

    # Check radius value and set to optimal, if not defined
    elif radius_x is None:
        logger.debug(
            "No radius value specified for disk kernel. Setting to maximal.")
        radius_x = (size - 1)/2.0

    # If no radius_y value has been specified, calculate circular kernel
    elif radius_y is None:
        radius_y = radius_x

    mu_x = (size_x - 1) / 2.0
    mu_y = (size_y - 1) / 2.0

    x, y = np.meshgrid(np.arange(0, size_x, 1), np.arange(0, size_y, 1))

    # Calculate mask, if the radii are not equal, results in an ellipse
    mask = (x-mu_x)**2/radius_x**2 + (y-mu_y)**2/radius_y**2 <= 1

    # Initialize kernel with zeros and apple mask.
    kernel = np.zeros((size_x, size_y))
    kernel[mask.T] = 1.0

    return images.normalize(kernel, normalize)


def get_kernel_gauss(size: int,
                     sigma: Optional[float] = None,
                     size_y: Optional[int] = None,
                     sigma_y: Optional[float] = None,
                     normalize: str = 'area') -> ndarray:
    """Get a normalized 2D Gauss kernel.

    This kernel is not a tensor product kernel, i.e. it is rotationally
    symmetric if the sigma values in x- and y-direction are equal.

    Args:
        size: Size of the kernel in pixels.

        sigma: Standard deviation of the Gauss kernel.
            Default: size/5.0.

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        sigma_y: Standard deviation of the Gauss kernel in v-direction.
            If no sigma_y has been specified, a symmetric kernel is calculated.

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A gauss kernel of shape (size, size_y) with radii standard deviations
        (sigma, sigma_y).

    """
    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size
    sigma_x = sigma

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x

    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    # Set kernel sigma values
    if sigma_x is None and sigma_y is None:
        sigma_x = min(size_x / 5.0, size_y / 5.0)
        sigma_y = sigma_x

    # Check sigma value and set to optimal, if not defined
    elif sigma_x is None:
        logger.debug(
            "No sigma value specified for Gauss kernel. Setting to optimal.")
        sigma_x = size_x / 5.0

    # If no sigma_y value has been specified, calculate symmetric gauss
    elif sigma_y is None:
        sigma_y = sigma_x

    logger.debug(f"Calculating Gauss kernel with shape ({size_x}, {size_y}).")

    # shift kernel zo image center
    mu_x = (size_x - 1) / 2.0
    mu_y = (size_y - 1) / 2.0

    x, y = np.meshgrid(np.arange(0, size_x, 1), np.arange(0, size_y, 1))
    kern = np.exp(
        -0.5*(((x-mu_x)**2)/(sigma_x**2) + ((y-mu_y)**2)/(sigma_y**2)))

    return images.normalize(kern.T, normalize)


def get_kernel_hann(size: int,
                    size_y: Optional[int] = None,
                    normalize: str = 'area') -> ndarray:
    """Get a normalized 2D Hann kernel.

    Args:
        size: Size of the kernel in pixels.

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A Hann kernel of shape (size, size_y).
    """

    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(f"Calculating Hann kernel with shape ({size_x}, {size_y}).")

    kern_x = np.hanning(size_x)
    kern_y = np.hanning(size_y)

    # Calculate outer product of the 1D kernels
    kern = np.outer(kern_x, kern_y)

    return images.normalize(kern, normalize)


def get_kernel_hamming(size,
                       size_y: Optional[int] = None,
                       normalize: str = 'area') -> ndarray:
    """Get a normalized 2D Hamming kernel.

    Args:
        size: Size of the kernel in pixels.

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A Hamming kernel of shape (size, size_y).
    """

    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(
        f"Calculating Hamming kernel with shape ({size_x}, {size_y}).")

    kern_x = np.hamming(size_x)
    kern_y = np.hamming(size_y)

    # Calculate outer product of the 1D kernels
    kern = np.outer(kern_x, kern_y)

    return images.normalize(kern, normalize)


def get_kernel_kaiser(size: int,
                      beta: float = 3.5,
                      size_y: Optional[int] = None,
                      beta_y: Optional[float] = None,
                      normalize: str = 'area') -> ndarray:
    """Get a normalized 2D Kaiser kernel.

    Args:
        size: Size of the kernel in pixels.

        beta: Shape parameter. Default: 3.5

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        beta_y: Shape parameter in y-direction.
            If no value is specified, a symmetric shape array is calculated

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A Kaiser kernel of shape (size, size_y).

    """

    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(f"Calculating Kaiser kernel with shape ({size_x}, {size_y}).")

    # Check beta value and set to optimal, if not defined
    if beta is None:
        logger.debug(
            "No beta value specified for Kaiser kernel. Setting to optimal.")
        beta = 3.5

    beta_x = beta

    # If no beta_y value has been specified, calculate symmetric gauss
    if beta_y is None:
        beta_y = beta_x

    kern_x = np.kaiser(size_x, beta_x)
    kern_y = np.kaiser(size_y, beta_y)

    # Calculate outer product of the 1D kernels
    kern = np.outer(kern_x, kern_y)

    return images.normalize(kern, normalize)


def get_kernel_hann_rotational(size: int,
                               size_y: Optional[int] = None,
                               normalize: str = 'area') -> ndarray:
    """Get a normalized, rotationally symmetric 2D Hann kernel.

    This kernel is rotationally symmetric and not a tensor product kernel.

    Args:
        size: Size of the kernel in pixels.

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A rotationally symmetric Hann kernel of shape (size, size_y).

    """

    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Set kernel size
    size_x = size

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(f"Calculating Hann kernel with shape ({size_x}, {size_y}).")

    distance_x = (size_x - 1) / 2
    distance_y = (size_y - 1) / 2
    distance = min(distance_x, distance_y)

    x_vector = np.arange(-distance_x, distance_x + 1)
    y_vector = np.arange(-distance_y, distance_y + 1)

    x_mesh, y_mesh = np.meshgrid(y_vector, x_vector)
    r = np.hypot(x_mesh, y_mesh)
    del x_mesh, y_mesh

    # Hanning window centralized around x = 0
    kern = 0.5*(1 - np.cos(np.pi*(r/distance + 1)))
    kern[np.where(r > distance)] = 0.0

    return images.normalize(kern, normalize)


def get_kernel_kaiser_rotational(size: int,
                                 beta: Optional[float] = None,
                                 size_y: Optional[int] = None,
                                 normalize: str = 'area') -> ndarray:
    """Get a normalized, rotationally symmetric 2D Kaiser kernel.

    This kernel is rotationally symmetric and not a tensor product kernel.
    Since the definition relies on the (recursively defined) modified
    bessel functions, this implementation is much slower than the tensor
    product implementation.

    Args:
        size: Size of the kernel in pixels.

        beta: Shape parameter. Default: 3.5

        size_y: Size of the kernel in y direction.
            If no value is specified, a quadratic kernel is calculated

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A rotationally symmetric Kaiser kernel of shape (size, size_y).

    """

    if type(size) is not int:
        raise ValueError(
            f"Specified size {size} is not an integer.")

    # Default value
    if beta is None:
        beta = 3.5

    # Set kernel size
    size_x = size

    # If no size_y value has been specified, calculate quadratic kernel
    if size_y is None:
        size_y = size_x
    else:
        if type(size_y) is not int:
            raise ValueError(
                f"Specified size_y {size_y} is not an integer.")

    logger.debug(f"Calculating Kaiser kernel with shape ({size_x}, {size_y}).")

    distance_x = (size_x - 1) / 2
    distance_y = (size_y - 1) / 2
    distance = min(distance_x, distance_y)

    x_vector = np.arange(-distance_x, distance_x + 1)
    y_vector = np.arange(-distance_y, distance_y + 1)

    x_mesh, y_mesh = np.meshgrid(y_vector, x_vector)
    r = np.hypot(x_mesh, y_mesh)
    del x_mesh, y_mesh

    # Kaiser window centralized around x = 0
    kern = iv(1, np.pi * beta * np.sqrt(1 - (r/distance)**2)) / iv(1, np.pi * beta)
    kern[np.where(r > distance)] = 0.0

    return images.normalize(kern, normalize)


# Wrapper function for all defined kernels
def get_kernel(name: str,
               size: int,
               size_y: Optional[int] = None,
               option: Optional[float] = None,
               option_y: Optional[float] = None,
               normalize: str = 'area'):
    """Get a predefined 2D matrix kernel

    Args:
        name: Name of the kernel. For a list of available kernel names, use
            :func:`get_avail_names()`.

        size: Size of the kernel in pixels.

        option: Option of the kernel.
            If ``disk`` is specified, corresponds to radius,
            if ``gauss`` is specified, corresponds to sigma,
            if ``kaiser`` is specified, corresponds to shape parameter beta.

        size_y: Size of the kernel in y-direction in pixels.
            If no size_y has been specified, a quadratic kernel is calculated.

        option_y: Option of the kernel in y-direcion..
            If ``disk`` is specified, corresponds to radius_y,
            if ``gauss`` is specified, corresponds to sigma_y,
            if ``kaiser`` is specified, corresponds to shape parameter beta_y.
            If no option_y has been specified,
            a symmetric kernel is calculated.

        normalize: Normalize the kernel by given method, see
            :func:`plenpy.utilities.images.normalize()` for available types.
            Default: Normalize to sum one.

    Returns:
        A predefined 2D kernel of specified type.

    """

    if name not in get_avail_names():
        raise ValueError(
            f"Specified argument name '{name}' is not one of the recognized "
            f"kernels: {get_avail_names()}")

    elif name == "disk":
        return get_kernel_disk(size=size,
                               radius=option,
                               size_y=size_y,
                               radius_y=option_y,
                               normalize=normalize)

    elif name == "gauss":
        return get_kernel_gauss(size=size,
                                sigma=option,
                                size_y=size_y,
                                sigma_y=option_y,
                                normalize=normalize)

    elif name == "hann":
        if option is not None or option_y is not None:
            logger.warning("Kernel 'hann' does not have option settings. "
                           "Parameter is ignored.")
        return get_kernel_hann(size=size,
                               size_y=size_y,
                               normalize=normalize)

    elif name == "hann_rotational":
        if option is not None or option_y is not None:
            logger.warning(
                "Kernel 'hann_rotational' does not have option settings. "
                "Parameter is ignored.")
        return get_kernel_hann_rotational(size=size,
                                          size_y=size_y,
                                          normalize=normalize)

    elif name == "hamming":
        if option is not None or option_y is not None:
            logger.warning("Kernel 'hamming' does not have option settings. "
                           "Parameter is ignored.")
        return get_kernel_hamming(size=size,
                                  size_y=size_y,
                                  normalize=normalize)

    elif name == "kaiser":
        return get_kernel_kaiser(size=size,
                                 beta=option,
                                 size_y=size_y,
                                 beta_y=option_y,
                                 normalize=normalize)

    elif name == "kaiser_rotational":
        return get_kernel_kaiser_rotational(size=size,
                                            beta=option,
                                            size_y=size_y,
                                            normalize=normalize)

    else:
        raise NotImplementedError(
            f"The specified kernel name '{name}' is not implemented.")


# Check symmetries
def is_symmetric(m: ndarray,
                 symmetry: Union[None, str] = None) -> bool:
    """ Utility function to check for symmetries for a NxN array.
        
    Args:
        m : Input ndarray to check.

    symmetry : Symmetry to check. Available symmetries are 'all', 'rotational'
        and 'axial'. If ``None`` is specified, all symmetries are checked.


    Returns:
        bool
            ``True`` if input is symmetric with the specified symmetry.
            ``False`` otherwise.
    """

    param_list = ["rotational", "axial"]

    if symmetry is None:
        return (is_symmetric(m, symmetry="axial")
                and is_symmetric(m, symmetry="rotational"))

    elif symmetry not in param_list:
        raise ValueError(
            f"Symmetry name '{symmetry}' invalid. "
            "Please specify a valid symmetry name.")

    elif symmetry == "axial":
        flip_lr = np.fliplr(m)
        flip_ud = np .flipud(m)

        if np.allclose(m, flip_lr, rtol=1e-9) \
                and np.allclose(m, flip_ud, rtol=1e-9):

            return True

        else:
            return False

    elif symmetry == "rotational":
        rot_90 = np.rot90(m)
        rot_180 = np.rot90(m, 2)
        rot_270 = np.rot90(m, 3)

        if np.allclose(m, rot_90, rtol=1e-9) \
                and np.allclose(m, rot_180, rtol=1e-9) \
                and np.allclose(m, rot_270, rtol=1e-9):

            return True

        else:
            return False
