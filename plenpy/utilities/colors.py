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
Module defining color conversion and representation.

"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from numpy.core.multiarray import ndarray
from scipy import interpolate

import plenpy.logg

logger = plenpy.logg.get_logger()

# Path to the color matching function and illuminant data
DATA_PATH = Path(pkg_resources.resource_filename("plenpy", "data/"))

# Available color matching functions and illuminants
# The data was extracted from the http://www.cvrl.org/ database of
# the  Institute of Ophthalmology at the University College London
CMFS = {
    # CIE 1931 standard observer, sampled at different wavelengths:
    "CIE_1931_1NM": "CMF_CIE1931_2deg_1nm.npy",
    "CIE_1931_5NM": "CMF_CIE1931_2deg_5nm.npy",
    #
    # Proposed CIE 2006 standard observer, sampled at different wavelengths:
    "CIE_2006_0_1NM": "CMF_CIE2006_2deg_0_1nm.npy",
    "CIE_2006_1NM": "CMF_CIE2006_2deg_1nm.npy",
    "CIE_2006_5NM": "CMF_CIE2006_2deg_5nm.npy",
    #
    # CIE 1931 standard observer, modified but Judd or Judd and Voss:
    "CIE_1931_JUDD_5NM": "CMF_CIE1931_JUDD_2deg_5nm.npy",
    "CIE_1931_JUDD_VOSS_5NM":  "CMF_CIE1931_JUDD_VOSS_2deg_5nm.npy"}

# CIE standard illuminants
ILLUMINANTS = {"CIE_A": "ILLUMINANT_CIE_A.npy",
              "CIE_D65": "ILLUMINANT_CIE_D65.npy"}

# CIE XYZ to RGB conversion matrix according to sRGB standard
XYZ_TO_RGB_MAT = np.asarray([[3.2404542, -1.5371385, -0.4985314],
                             [-0.9692660, 1.8760108, 0.0415560],
                             [0.0556434, -0.2040259, 1.0572252]])


def get_avail_cmfs() -> List[str]:
    """Get a list of available color matching functions.

    """
    return [key for key in CMFS]


def get_avail_illuminants() -> List[str]:
    """Get a list of available illuminants.

    """
    return [key for key in ILLUMINANTS]


def get_cmf(cmf: str = "CIE_1931_1NM") -> ndarray:
    """Get a color matching function.

    Args:
        cmf: Name of the color matching function. See :func:`get_avail_cmfs()`
            For a list of available color matching functions.
            Default: CIE 1931 standard observer sampled at 1nm.

    Returns:
        Color matching function of shape (num_wavelengths, 4) in format
        [wavelength, x, y, z].

    """
    if cmf not in get_avail_cmfs():
        raise ValueError(f"The color matching function '{cmf}' is not one of "
                         f"the available functions {get_avail_cmfs()}.")
    filename = DATA_PATH / CMFS[cmf]
    return np.load(filename)


def get_illuminant(illuminant: str = "CIE_D65") -> ndarray:
    """Get an illuminant.

    Args:
        illuminant: Name of the illuminant. See :func:`get_avail_illuminants()`
            For a list of available illuminants.
            Default: CIE D65 midday light illuminant.

    Returns:
        Illuminant of shape (num_wavelengths, 4) in format
        [wavelength, x, y, z].

    """
    if illuminant not in get_avail_illuminants():
        raise ValueError(f"The illuminant '{illuminant}' is not one of the "
                         f"available illuminants {get_avail_illuminants()}.")
    filename = DATA_PATH / ILLUMINANTS[illuminant]
    return np.load(filename)


def xyz_to_rgb(xyz: Union[ndarray, List[float]]) -> ndarray:
    """Convert XYZ value to RGB.

    The conversion of CIE XYZ values to RGB is done using the sRGB standard.
    RGB values are bound by [0, 1] while XYZ values are bound by [0, 100].

    Args:
        xyz: Color values [x, y, z].

    Notes:
        If multiple values are specified, they must be vertically stacked,
        i.e. xyz = [[x1, y1, z1], [x2, y2, z2]]
    """
    if not (type(xyz) == list or type(xyz) == ndarray):
        raise ValueError(
            "Please specify a valid XYZ value as a list or ndarray.")

    xyz_nd = np.asarray(xyz)
    if xyz_nd.shape[-1] != 3:
        raise ValueError("The passed XYZ values have to be of length 3. "
                         "If You have specified multiple XYZ values, they"
                         "must be stacked vertically.")

    return np.clip(XYZ_TO_RGB_MAT @ xyz_nd.T / 100, a_min=0, a_max=1).T


def rgb_to_xyz(rgb: Union[ndarray, List[float]]) -> ndarray:
    """Convert RGB value to XYZ.

    The conversion of RGB values to CIE XYZ is done using the sRGB standard.
    RGB values are bound by [0, 1] while XYZ values are bound by [0, 100].

    Args:
        rgb: Color values [r, g, b].

    Notes:
        If multiple values are specified, they must be vertically stacked,
        i.e. xyz = [[r1, g1, b1], [r2, g2, b2]]

    """
    if not (type(rgb) == list or type(rgb) == ndarray):
        raise ValueError(
            "Please specify a valid RGB value as a list or ndarray.")

    rgb_nd = np.asarray(rgb)
    if rgb_nd.shape[-1] != 3:
        raise ValueError("The passed RGB values have to be of length 3. "
                         "If You have specified multiple RGB values, they"
                         "must be stacked vertically.")

    return (100.0*np.linalg.inv(XYZ_TO_RGB_MAT) @ rgb_nd.T).T


class Converter(ABC):
    """Abstract conversion class.

    This class is a base class for wavelength or spectrum conversion
    using CIE standard observer and illuminants.

    """

    def __init__(self,
                 cmf: str,
                 illuminant: str):
        """:class:`Converter` base class initialization.

        Args:
            cmf: Name of the color matching function.
                See :func:`get_avail_cmfs()` for a list of available color
                matching functions.
                Default: CIE 1931 standard observer sampled at 1nm.

            illuminant: Name of the illuminant. See
                :func:`get_avail_illuminants()` for a list of available
                illuminants.
                Default: CIE D65 midday light illuminant.

        """
        self.cmf_name = cmf
        self.illuminant_name = illuminant

        # Get illuminant and interpolate
        illuminant = get_illuminant(illuminant)
        self._lum_interp = interpolate.interp1d(illuminant[:, 0],
                                                illuminant[:, 1],
                                                kind='cubic')

        # Get color matching functions and interpolate
        cmf = get_cmf(cmf)

        wavelengths_cmf_base = cmf[:, 0]
        x_base = cmf[:, 1]
        y_base = cmf[:, 2]
        z_base = cmf[:, 3]

        self._x_interp = interpolate.interp1d(cmf[:, 0], cmf[:, 1],
                                             kind='cubic')
        self._y_interp = interpolate.interp1d(cmf[:, 0], cmf[:, 2],
                                             kind='cubic')
        self._z_interp = interpolate.interp1d(cmf[:, 0], cmf[:, 3],
                                             kind='cubic')

        # Minimum and maximum wavelengths of cmf and illuminant
        self._lambda_xyz_min = cmf[0, 0]
        self._lambda_xyz_max = cmf[-1, 0]

        self._lambda_lum_min = illuminant[0, 0]
        self._lambda_lum_max = illuminant[-1, 0]

        self._x_vals = None
        self._y_vals = None
        self._z_vals = None

        self._lum_vals = None
        self._lum_normalization = None

        return

    @abstractmethod
    def to_xyz(self, **kwargs) -> ndarray:
        pass

    @abstractmethod
    def to_rgb(self, **kwargs) -> ndarray:
        pass


class SpectrumConverter(Converter):
    """Spectrum conversion class.

    This class is used to compute XYC or RGB values from an EM spectrum.
    The conversion can be done with different standard observers. For a list
    of available color matching functions, see :func:`get_avail_cfs()`.
    To calculate XYZ or RGB values for reflectance spectra, an illuminant is
    needed. For a list of available illuminants,
    see :func:`get_avail_illuminants()`.

    Attributes:
        wavelengths: Wavelengths basis of the spectra in nanometers.

    """
    def __init__(self,
                 wavelengths: Union[ndarray, List[float]],
                 cmf: str = "CIE_1931_1NM",
                 illuminant: str = "CIE_D65"):
        """:class:`SpectrumConverter` class initialization.

        Args:
            wavelengths: Wavelengths basis of the spectra in nanometers.
                For example [400, 450, 500, 550, 600, 650, 700].

            cmf: Name of the color matching function.
                See :func:`get_avail_cmfs()` for a list of available color
                matching functions.
                Default: CIE 1931 standard observer sampled at 1nm.

            illuminant: Name of the illuminant. See
                :func:`get_avail_illuminants()` for a list of available
                illuminants.
                Default: CIE D65 midday light illuminant.

        """
        # Init base class
        super().__init__(cmf=cmf, illuminant=illuminant)

        # Check the wavelength edges of passed spectrum and cmf, illuminant
        # We need to find the joint union of all wavelength bases.
        # Save the crop mask, as all passed spectra will need to be cropped
        self._crop_mask = np.logical_and(
            wavelengths >= max(self._lambda_lum_min, self._lambda_xyz_min),
            wavelengths <= min(self._lambda_lum_max, self._lambda_xyz_max)
        )
        self._wave_basis_length = len(wavelengths)
        self.wavelengths = wavelengths[self._crop_mask]

        # Get interpolation values
        self._x_vals = self._x_interp(self.wavelengths)
        self._y_vals = self._y_interp(self.wavelengths)
        self._z_vals = self._z_interp(self.wavelengths)
        self._lum_vals = self._lum_interp(self.wavelengths)

        # Calculate luminance normalization
        self._lum_normalization = 100.0 / (self._lum_vals @ self._y_vals)

        return

    def to_xyz(self, spectrum: Union[ndarray, List[float]]) -> ndarray:
        """Convert spectrum to CIE XYZ values.

        Passing multiple spectra is possible.

        Args:
            spectrum: Spectrum or vertically stacked collection of spectra.
                The length of the spectra have to match the length of the
                converter's wavelength basis.

        Returns:
            CIE XYZ values of the spectrum.

        Note:
            If multiple spectra are specified, they must be vertically stacked,
            i.e. xyz = [[spectrum1], [spectrum2]]. Then the result will be
            the vertically stacked XZY values [[x1, y1, z1], [x2, y2, z2]].

        """
        # Calculate mask from crop_mask to be applied to multiple spectra
        # Basically, vertically stack copies of the crop_mask.
        spectrum = np.asarray(spectrum)
        num_specs = 1 if spectrum.ndim == 1 else spectrum.shape[0]

        length = len(spectrum) if num_specs == 1 else spectrum.shape[1]
        if length != self._wave_basis_length:
            raise ValueError(
                "The specified spectrum has to have the same length as the "
                "wavelength basis that the converter was initialized with.")

        mask = np.reshape(
            np.tile(self._crop_mask, reps=num_specs),
            spectrum.shape)

        # Crop the spectrum. Squeeze is needed if only one spectrum is passed.
        cropped_spectrum = np.squeeze(
            np.reshape(spectrum[mask],
                       (num_specs, self.wavelengths.shape[0])))

        # Integrate to get XYZ value. Intervall width is not needed here as
        # it will cancel with the one in self._lum_normalization ...
        x = cropped_spectrum @ self._x_vals
        y = cropped_spectrum @ self._y_vals
        z = cropped_spectrum @ self._z_vals

        return self._lum_normalization*np.asarray([x, y, z]).T

    def to_rgb(self, spectrum: Union[ndarray, List[float]]) -> ndarray:
        """Convert spectrum to RBG values.

        This functions converts the spectra to CIE XYZ values and then converts
        to RGB via the sRGB standard.

        Passing multiple spectra is possible.

        Args:
            spectrum: Spectrum or vertically stacked collection of spectra.
                The length of the spectra have to match the length of the
                converter's wavelength basis.

        Returns:
            RGB values of the spectrum according to sRGB standard.

        Note:
            If multiple spectra are specified, they must be vertically stacked,
            i.e. xyz = [[spectrum1], [spectrum2]]. Then the result will be
            the vertically stacked XZY values [[x1, y1, z1], [x2, y2, z2]].

        """

        return xyz_to_rgb(self.to_xyz(spectrum))


class WavelengthConverter(SpectrumConverter):

    def __init__(self,
                 cmf: str = "CIE_1931_1NM",
                 illuminant: str = "CIE_D65"):
        """:class:`WavelengthConverter` class initialization.

        Args:
            cmf: Name of the color matching function.
                See :func:`get_avail_cmfs()` for a list of available color
                matching functions.
                Default: CIE 1931 standard observer sampled at 1nm.

            illuminant: Name of the illuminant. See
                :func:`get_avail_illuminants()` for a list of available
                illuminants.
                Default: CIE D65 midday light illuminant.
        """
        # Create SpectrumConverter with spectrum from 360nm to 830nm
        super().__init__(wavelengths=np.arange(400, 700, 1),
                         cmf=cmf,
                         illuminant=illuminant)
        return

    def to_xyz(self,
               wavelength: Union[float, List[float]],
               var: float = 1500):
        """Convert a wavelength to an approximate CIE XYZ value.

        Converting is done by calculating a Gaussian shaped spectrum with
        central wavelength of the passed wavelength and converting the
        spectrum to XYZ value via CIE standard observer and illuminant,
        see also :class:`SpectrumConverter`.

        Args:
            wavelength: Central wavelength.

            var: Variance of the Gaussian used to calculate the spectrum.
                Note: Spectra that are too small, e.g. "monochromatic" light,
                will result in a near to black XYZ value. Therefore, the
                default value is chosen rather large.

        Returns:
            Approximate XYZ value of the wavelength.

        Note:
            If multiple wavelengths are specified as a list,
            i.e. [400, 467, 555], then the result will be
            the vertically stacked XZY values [[x1, y1, z1], [x2, y2, z2]].

        """
        # Convert to array
        wavelenghts = np.asarray([wavelength])
        num_waves = 1 if wavelenghts.ndim == 1 else wavelenghts.shape[1]

        # Create spectrum from provided wavelength
        spectrum = np.exp(-(self.wavelengths - wavelenghts.T)**2/(2 * var))

        return super().to_xyz(spectrum)

    def to_rgb(self,
               wavelength: Union[float, List[float]],
               var: float = 1500):
        """Convert a wavelength to an approximate RGB value.

        Converting is done by calculating a Gaussian shaped spectrum with
        central wavelength of the passed wavelength and converting the
        spectrum to XYZ value via CIE standard observer and illuminant, and
        conversion to RGB via the sRGB standard,
        see also :class:`SpectrumConverter`.

        Args:
            wavelength: Central wavelength.

            var: Variance of the Gaussian used to calculate the spectrum.
                Note: Spectra that are too small, e.g. "monochromatic" light,
                will result in a near to black RGB value. Therefore, the
                default value is chosen rather large.

        Returns:
            Approximate RGB value of the wavelength.

        Note:
            If multiple wavelengths are specified as a list,
            i.e. [400, 467, 555], then the result will be
            the vertically stacked XZY values [[r1, g1, b1], [r2, g2, b2]].

        """

        return xyz_to_rgb(self.to_xyz(wavelength=wavelength, var=var))


def show_beautiful_rainbow(lambda_start: int = 350,
                           lambda_end: int = 800,
                           size:int = 1000):
    """Plots a nice rainbow gradient picture.

    This visualizes the wavelength to spectrum to RGB conversion as implemented
    by :class:`WavelengthConverter` and :class:`SpectrumConverter`.

    Args:
        lambda_start: Start wavelength.

        lambda_end: End wavelength.

        size: Size of the picture in pixels.

    """
    converter = WavelengthConverter(cmf='CIE_1931_1NM', illuminant='CIE_D65')

    waves = np.linspace(lambda_start, lambda_end, size)
    img = np.ones((300, len(waves), 3))

    for i in range(0, len(waves)):
        rgb = converter.to_rgb(waves[i])

        img[:, i, 0] *= rgb[0]
        img[:, i, 1] *= rgb[1]
        img[:, i, 2] *= rgb[2]

    ticklabels = np.linspace(lambda_start, lambda_end, 10)
    ticks = np.linspace(0, len(waves), 10)
    fig = plt.figure()
    plt.yticks([])
    plt.xticks(ticks, ticklabels)
    plt.xlabel("Wavelength (nm)")
    plt.imshow(img)
    plt.show()

    return
