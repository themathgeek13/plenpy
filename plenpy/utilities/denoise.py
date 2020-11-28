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
Module defining various image denoising algorithms

"""
from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray
from scipy import interpolate

import plenpy.logg

logger = plenpy.logg.get_logger()


def remove_outliers(img: ndarray,
                    confidence: ndarray,
                    threshold: float = 0.5) -> ndarray:
    """Filter an image by removing outliers with confidence below a threshold.

    Pixel with confidence below the threshold are discarded and are
    recalculated by interpolation from the surrounding pixels.

    ..todo: what happens on image borders?

    Args:
        img: Input image. Has to be of the shape (x,y) with a single color channel.

        confidence: Confidence measure of image pixels.
            Needs to have same shape as input image.

        threshold: Pixel with confidence below threshold are discarded.

    Returns:
        The filtered image of the same shape as the input image.
    """

    if img.shape != confidence.shape:
        raise ValueError(f"Input image {img.shape} and confidence "
                         f"{confidence.shape} do not have the same shape.")
    if img.ndim != 2:
        raise ValueError(f"Invalid image dimension {img.ndim}. "
                         "Must have dimension 2")

    # Interpolation grid
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Find outliers based on confidence map and threshold
    index_good = np.where(confidence > threshold, 1, 0)

    # Interpolate to fill holes of removed outliers
    img_filtered = interpolate.griddata((xx[index_good == 1], yy[index_good == 1]),
                                        img[index_good == 1],
                                        (xx.ravel(), yy.ravel()),
                                        method='linear')

    return img_filtered.reshape(img.shape)


def denoise_image_rof(img: ndarray,
                      n_iter: int,
                      w_lambda: Union[ndarray, float] = 5) -> ndarray:

    """Primal-dual algorithm for minimization of ROF-functional (TV-L2).
    Fast form of primal-dual algorithm (faster than standard).
    Algotirhm is based on [R1]_.

    Discrete functional:

    - ROF:   min_x( ||_nabla x||_1 + 0.5*lambda*(||x - f||_1)**2 )

    Args:
        img:
            Input image.

        n_iter:
            Number of iterations

        w_lambda:
            Weight factor of data term. Pixel-wise weight is possible.

    Returns:
        The filtered image of the same shape as the input image.

    References:
        .. [R1] Chambolle, Antonin; Pock, Thomas (2011): A First-Order
           Primal-Dual Algorithm for Convex Problems with Applications
           to Imaging. In: Journal of Mathematical Imaging and Vision 40 (1)
    """

    u = img.copy()
    y = _nabla(u)

    L2 = 8
    tau = 0.02
    sigma = 1.0/(L2*tau)
    gamma = 0.35*w_lambda

    for i in range(n_iter):

        # Calculate gradient
        u_grad = _nabla(u)

        # Optimize dual variable ( prox_f )
        y = y + sigma*u_grad

        # Projection
        y = _prox_tv(y)

        # Optimize primal variable ( prox_g )
        u_new = u - tau * _nablaT(y)

        # l2-norm (ROF denoising)
        u_new = _prox_l2(u_new, img, w_lambda * tau)

        # Optimize step-size (faster convergence)
        theta = 1/np.sqrt(1 + 2*gamma*tau)
        tau = theta*tau
        sigma = sigma/theta

        # Extrapolate
        u = u_new + theta*(u_new - u)

        # Break if max accuracy reached
        # if (np.abs(u[:]-u_new[:])).sum() < tol:
        #     print(i)
        #     break

    return u


def denoise_image_tvl1(img: ndarray,
                       n_iter: int,
                       w_lambda: Union[ndarray, float] = 0.5) -> ndarray:
    """Primal-dual algorithm for minimization of TV-L1-functional.
    Algotirhm is based on [R1]_.

    Discrete functional:

    - TV-L1: min_x( ||_nabla x||_1 + lambda*||x - f||_1 )

    Args:
        img:
            Input image.

        n_iter:
            Number of iterations

        w_lambda:
            Weight factor of data term. Pixel-wise weight is possible.

    Returns:
        The filtered image of the same shape as the input image.

    """

    u = img.copy()
    y = _nabla(u)

    L2 = 8.0
    tau = 0.02
    sigma = 1.0/(L2*tau)
    theta = 1.0

    # Iterative primal-dual algorithm
    for i in range(n_iter):
        # Calculate gradient
        u_grad = _nabla(u)

        # Optimize dual variable ( prox_f ) TV
        y = y + sigma*u_grad
        # Projection
        y = _prox_tv(y)

        # Optimize primal variable ( prox_g )
        u_new = u - tau * _nablaT(y)
        # l1-norm (shrink) (TV-l1 denoising)
        u_new = _prox_l1(u_new, img, w_lambda * tau)

        # Extrapolate
        u = u_new + theta*(u_new - u)

        # Break if max accuracy reached
        # if (np.abs(u[:]-u_new[:])).sum() < tol:
        #     print(i)
        #     break

    return u


def denoise_image_huber(img: ndarray,
                       n_iter: int,
                       w_lambda: Union[ndarray, float] = 0.5) -> ndarray:
    """Primal-dual algorithm for minimization of TV-Huber-norm-L1-functional.
    Algotirhm is based on [R1]_.

    Discrete functional:

    - TV_Huber-L1: min_x( ||_nabla x||_h + lambda*||x - f||_1 )

    Args:
        img:
            Input image.

        n_iter:
            Number of iterations

        w_lambda:
            Weight factor of data term. Pixel-wise weight is possible.

    Returns:
        The filtered image of the same shape as the input image.

    """
    L2 = 8.0
    alpha = 0.05
    gamma = 5
    delta = alpha

    mu = 2*np.sqrt(gamma*delta)/np.sqrt(L2)
    tau = mu/2/gamma
    theta = 1/(1 + mu)
    sigma = mu/2/delta

    # Iterative primal-dual algorithm
    u = img.copy()
    y = _nabla(u)
    for i in range(n_iter):
        # Optimize dual variable ( prox_f ) TV
        y = y + sigma * _nabla(u)

        # Projection (TV with huber norm)
        y = _prox_tv(y, 1 + sigma * alpha) / (1 + sigma * alpha)

        # Optimize primal variable ( prox_g )
        u_new = u - tau * _nablaT(y)

        # l1-norm (shrink)
        u_new = _prox_l1(u_new, img, w_lambda * tau)

        # Extrapolate
        u = u_new + theta*(u_new - u)

        # Break if max accuracy reached
        # if (np.abs(u[:]-u_new[:])).sum() < tol:
        #     print(i)
        #     break

    return u


def denoise_multi_images_tvl1(imgs: Union[ndarray, list],
                              n_iter: int,
                              w_lambdas: Union[ndarray, float] = 0.5) -> ndarray:
    """Primal-dual algorithm for minimization of TV-L1 functional with huber norm
    in variational term and several weights for fusion of multiple measurements.
    Algotirhm is based on [R1]_.

    Discrete functional:

    - min_x( ||_nabla x||_h + sum_i{ lambda_i*||x - f_i||_1 } )

    Args:
        imgs:
            Stack of input images.

        n_iter:
            Number of iterations

        w_lambdas:
            Weight factor of the multiple data terms. Pixel-wise weight is possible.

    Returns:
        The filtered image of the same shape as the input image.

    """

    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2 * tau)
    theta = 1.0

    imgs = np.asarray(imgs)
    u = np.nanmean(imgs, axis=2)
    y = _nabla(u)
    r = np.zeros(imgs.shape)

    alpha = 0.05

    # Iterative primal-dual algorithm
    for i in range(n_iter):
        # Calculate gradient
        u_grad = _nabla(u)

        # Optimize dual variable ( prox_f ) TV
        y = y + sigma*u_grad
        # Projection
        y = _prox_tv(y, 1 + sigma * alpha) / (1 + sigma * alpha) # Huber Norm with total variation

        r = np.clip(r + sigma*(u[..., np.newaxis] - imgs), -w_lambdas, w_lambdas)

        # Optimize primal variable ( prox_g )
        u_new = u - tau*(_nablaT(y) + np.nansum(r, axis=-1))

        # Extrapolate
        u = u_new + theta*(u_new - u)

    return u


def _nabla(img_in: ndarray) -> ndarray:
    """ Calculates gradient of input image.

    Args:
        img_in: Input image (h,w).
    Returns:
        Gradient of input image (h,w,2).
    """
    h, w = img_in.shape
    img_out = np.zeros((h, w, 2), img_in.dtype)
    img_out[:, :-1, 0] -= img_in[:, :-1]
    img_out[:, :-1, 0] += img_in[:, 1:]
    img_out[:-1, :, 1] -= img_in[:-1, :]
    img_out[:-1, :, 1] += img_in[1:, :]

    return img_out


def _nablaT(img_in: ndarray) -> ndarray:
    """ 'Inverse' to _nabla. Calculates divergence of image. Input is gradient image.

    Args:
        img_in: Input gradient image (h,w,2).
    Returns:
        Divergence of image (h,w).
    """
    h, w = img_in.shape[:2]
    img_out = np.zeros((h, w), img_in.dtype)
    img_out[:, :-1] -= img_in[:, :-1, 0]
    img_out[:, 1:] += img_in[:, :-1, 0]
    img_out[:-1, :] -= img_in[:-1, :, 1]
    img_out[1:, :] += img_in[:-1, :, 1]
    return img_out


def _prox_l1(u: ndarray,
             u_ref: ndarray,
             lt: float) -> ndarray:
    """
    Proximity operator for l1-norm. Pixel-wise "shrinking".

    Args:
        u:
            Image to optimize.
        u_ref:
            Original image.
        lt:
            lambda*tau from primal-dual algorithm.
    Returns:
        Optimized image.
    """
    u = (u - lt) * (u - u_ref > lt) + \
        (u + lt) * (u - u_ref < -lt) + \
        u_ref * (np.abs(u - u_ref) <= lt)
    return u


def _prox_l2(u: ndarray,
             u_ref: ndarray,
             lt: float) -> ndarray:
    """
    Proximity operator for l2-norm.

    Args:
        u:
            Image to optimize.
        u_ref:
            Original image.
        lt:
            lambda*tau from primal-dual algorithm.
    Returns:
        Optimized image.
    """
    u = (u + lt*u_ref)/(1 + lt)
    return u


def _prox_tv(y: ndarray,
             r: float = 1.0) -> ndarray:
    """
    Proximity operator for total variation. Pixel-wise "projection onto r-balls".

    Args:
        y:
            Dual variable from primal-dual algorithm.
        r:
            Radius of ball.
    Returns:
        Optimized variable.
    """
    norm_y = np.maximum(1, np.sqrt((y*y).sum(-1))/r)
    y = y/norm_y[..., np.newaxis]
    return y
