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
Module defining plotting methods.

This module is a collection of different plotting functions that are regularly
needed in development or research. The plots are made to be of publish quality.

TODO: Implement PGF or TikZ export options.

"""
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.core.multiarray import ndarray

import plenpy.logg
from plenpy.utilities import images

logger = plenpy.logg.get_logger()


def plot_fft(img: ndarray,
             angle=False,
             shift: bool = True,
             window: Optional[str] = None,
             pad: Optional[int] = None,
             implementation: str = 'scipy',
             interpolation: Union[None, str] = None,
             cmap: str = 'jet',
             vmin: float = 1e-6,
             vmax: Optional[float] = 1,
             norm: bool = True,
             rescale: bool = True,
             plt_show: bool = True):
    """Utility function to plot the absolute value of the 2D fourier transform
    of an image.

    Args:
        img: Input image.

        angle: Boolean flag indicating whether to also, seperately,
            plot the angle part of FFT.

        interpolation: Specify interpolation mode of the plotted image.
            Available interpolations are 'bilinear' and 'bicubic'.
            If ``None``, default to rc ``image.inerpolation`` of matplotlib
            (i.e., usually no interpolation).

        cmap: Color map used for plotting.

        vmin: Minimum value to plot.

        vmax: Maximum value to plot.

        norm: Whether to normalize the FFT by its size.

        rescale: Specify whether to rescale the axis from pixel to frequency
            domain.

        plt_show: Flag indicating whether to call
            :func:`.matplotlib.pyplot.show` at the end.

    For the remaining arguments, see :func:`plenpy.images.fourier`.

    """
    logger.info(f"Calculating and plotting Fourier transform "
                f"using {implementation}...")

    res = images.fourier(img,
                         shift=shift,
                         window=window,
                         pad=pad,
                         implementation=implementation)

    plot_title = f"Absolute value of Fourier transform"

    shift_x = 1 / (2 * res.shape[0])
    shift_y = 1 / (2 * res.shape[1])

    if shift:
        plot_title += f" (shifted)"

        # Shift zero component to middle
        extent = [-0.5 - shift_y,
                  0.5 + shift_y,
                  0.5 + shift_x,
                  -0.5 - shift_x]
    else:
        extent = [0 - shift_y,
                  1 + shift_y,
                  1 + shift_x,
                  0 - shift_x]

    xlabel = "f_y in 1/px"
    ylabel = "f_x in 1/px"

    if not rescale:
        extent = None
        xlabel = "AU"
        ylabel = "AU"

    fig, ax = plt.subplots()

    abs = np.abs(res)
    if norm:
        abs /= abs.size

    fig.suptitle(plot_title)
    im = ax.imshow(abs,
                   aspect='equal',
                   interpolation=interpolation,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap=cmap,
                   extent=extent)
    fig.colorbar(im)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if angle:
        fig_angle, ax_angle = plt.subplots()
        fig_angle.suptitle("Angle of Fourier transform")
        im = ax_angle.imshow(np.angle(res),
                             aspect='equal',
                             interpolation=interpolation,cmap=cmap,
                             extent=extent)
        ax_angle.set_xlabel(xlabel)
        ax_angle.set_ylabel(ylabel)
        fig_angle.colorbar(im)

    if plt_show is True:
        plt.show()

    logger.info("...done")
    return
