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

"""Module defining the core :class:`SpectralArray` class.
This is the base class for the light field and spectral image classes.

"""

import re
from pathlib import Path
from typing import List, Optional, Union, Type, Any, Tuple

import h5py
import numpy as np
import scipy.io
from numpy.core.multiarray import ndarray
from scipy.signal import decimate, resample
from skimage.util.dtype import convert

import plenpy.logg
from plenpy.utilities import colors
from plenpy.utilities import images

logger = plenpy.logg.get_logger()

_all_ = ['BandInfo', 'SpectralArray', 'DimensionError']


class BandInfo:
    """A class holding the band information of a spectral image.

    The band information characterizes the spectral properties of every color
    channel of the image.

    Every list attribute has the be of length N = ``_num_ch``

    Attributes:
        _numChannels (int): Number of spectral channels.

        _centers (Union[List[float], ndarray]): List of central wavelengths
                                                of each channel.

        _unit (str): Unit of the wavelenghts, e.g. ``nm``.

        _bandwidths (Union[List[float], ndarray]): List of bandwidths of each
                                                   channel. Optional.

        _centers_std (Union[List[float], ndarray]): List of standard deviation
                                                    of band centers. Optional.

        _bandwidths_std (Union[List[float], ndarray]): List of standard
                                                       deviation of bandwidths.
                                                       Optional.

        _type (str): Imgae type, e.g. ``reflectance``. Optional.


    """

    def __init__(self,
                 num_channels: int,
                 centers: Union[List[float], ndarray],
                 unit: str,
                 bandwidths: Optional[Union[List[float], ndarray]] = None,
                 centers_std: Optional[Union[List[float], ndarray]] = None,
                 bandwidths_std: Optional[Union[List[float], ndarray]] = None,
                 type: Optional[str] = None):
        self._numChannels = num_channels

        def check_length(input_list: Union[list, ndarray]):
            """Check length of input list/array.
            """
            if input_list is not None and len(input_list) != num_channels:
                raise ValueError(
                    "Length of list has to match number of channels")

        check_length(centers)
        self._centers = centers

        check_length(bandwidths)
        self._bandWidths = bandwidths

        check_length(centers_std)
        self._centersStd = centers_std

        check_length(bandwidths_std)
        self._bandWidthsStd = bandwidths_std

        self._type = type
        self._unit = unit

    @classmethod
    def from_equidistant(cls,
                         num_channels: int,
                         lambda_start: int,
                         lambda_end: int,
                         bandwidth: Optional[float] = None,
                         center_std: Optional[float] = None,
                         bandwidth_std: Optional[float] = None,
                         type: Optional[str] = None):
        """Initialize a BandInfo object with equidistant wavlength samples.

        Args:
            num_channels: Number of wavelength channels.

            lambda_start: Wavelength range start in nanometers.

            lambda_end: Wavelength range end in nanometers.

            bandwidth: Bandwith of channels. Optional.

            center_std: Standard deviation of centers. Optional.

            bandwidth_std: Standard deviation of bandwiths. Optional.

            type: Imgae type, e.g. ``reflectance``. Optional.
        """

        centers = np.linspace(lambda_start, lambda_end, num_channels)
        unit = 'nm'

        def get_list(input_float):
            if input_float is not None:
                return [input_float for i in range(num_channels)]
            else:
                return None

        return cls(num_channels=num_channels, centers=centers, unit=unit,
                   bandwidths=get_list(bandwidth), type=type,
                   centers_std=get_list(center_std),
                   bandwidths_std=get_list(bandwidth_std))

    @classmethod
    def from_envi_header(cls, hdr: dict):
        """Initialize a BandInfo object from a ENVI header dictionary.

        Args:
            hdr: Input dictionary from ENVI header file.

        """
        num_ch = int(hdr['bands'])

        centers_std = None      # ENVI does not provide standard deviation
        bandwiths_std = None    # ENVI does not provide standard deviation
        type = None             # ENVI does not provide a type

        try:
            centers = np.asarray(hdr['wavelength']).astype(np.float)
        except KeyError:
            centers = None

        try:
            unit = str(hdr['wavelength units'])
        except KeyError:
            unit = None

        try:
            bandwiths = np.asarray(hdr['fwhm']).astype(np.float)
        except KeyError:
            bandwiths = None

        return cls(num_channels=num_ch,
                   centers=centers,
                   unit=unit,
                   bandwidths=bandwiths,
                   type=type,
                   centers_std=centers_std,
                   bandwidths_std=bandwiths_std)

    def __eq__(self, other: 'BandInfo') -> bool:
        """Equality operator for BandInfo class.

        Args:
            other: BandInfo object to test equality with.

        Returns:
            ``True``, if BandInfo's are equal, ``False`` otherwise.

        """
        res = False

        # Check equality of all members explicitly
        if np.array_equal(self.num_channels, other.num_channels) and \
                np.array_equal(self.centers, other.centers) and \
                np.array_equal(self.bandwidths, other.bandwidths) and \
                np.array_equal(self.bandwidths_std, other.bandwidths_std) and \
                np.array_equal(self.centers_std, other.centers_std) and \
                self.type == other.type and \
                self.unit == other.unit:
            res = True

        return res

    @property
    def centers(self):
        return self._centers

    @property
    def bandwidths(self):
        return self._bandWidths

    @property
    def centers_std(self):
        return self._centersStd

    @property
    def bandwidths_std(self):
        return self._bandWidthsStd

    @property
    def num_channels(self):
        return self._numChannels

    @property
    def type(self):
        return self._type

    @property
    def unit(self):
        return self._unit

    def copy(self) -> 'BandInfo':
        """Get copy of the object.

        Returns:
            Copy of BandInfo object.
        """
        num_channels = self._numChannels
        centers = self._centers.copy()
        bandwidths = self._bandWidths.copy()
        centers_std = self._centersStd.copy()
        bandwidth_std = self._bandWidthsStd.copy()
        type = self.type
        unit = self.unit.lower()

        return BandInfo(num_channels=num_channels, centers=centers,
                        bandwidths=bandwidths, centers_std=centers_std,
                        bandwidths_std=bandwidth_std, type=type, unit=unit)

    def get_downsampled(self, num: int):
        """Get a copy of the BandInfo object corresponding to the downsampled
        spectral array.

        Note that equidistant sampling is assumed.
        For irregularly sampled SpectralArrays, the downsampled BandInfo
        will be incorrect.

        Args:
            num: Number of samples in the downsampled object.

        Returns:
            BandInfo object of the downsampled SpectralArray.

        """

        lambda_start = self.centers[0]
        lambda_end = self.centers[-1]
        band_info_out = BandInfo.from_equidistant(num,
                                                  lambda_start=lambda_start,
                                                  lambda_end=lambda_end,
                                                  type=self.type)

        return band_info_out

    def get_subband(self, start: int, stop: int):
        """Get a spectral subband of the BandInfo object.

        The subband is specified by a start index and stop index which is not
        included as is usual for Python array indexing x[start:stop].

        Args:
            start: Index of the start wavelength.

            stop: Index of the stop wavelength (not included).

        Returns:
            BandInfo object of the subband SpectralArray.

    """
        centers = self.centers[start:stop]
        num_ch = centers.size

        bandwidths = None
        if self.bandwidths is not None:
            bandwidths = self.bandwidths[start:stop]

        centers_std = None
        if self.centers_std is not None:
            centers_std = self.centers_std[start:stop]

        bandwidths_std = None
        if self.bandwidths_std is not None:
            bandwidths_std = self.bandwidths_std[start:stop]

        band_info_out = BandInfo(num_channels=num_ch,
                                 centers=centers,
                                 unit=self.unit,
                                 bandwidths=bandwidths,
                                 centers_std=centers_std,
                                 bandwidths_std=bandwidths_std,
                                 type=self.type)

        return band_info_out


class DimensionError(Exception):
    """Error can be raised by inherited classes when expected dimension does
    not match the dimension of the passed array."""

    def __init__(self, exp: int, found: int):
        self.exp = exp
        self.found = found

    def __str__(self):
        return f"Expected {self.exp}D input. Found {self.found}D."


class SpectralArray(np.ndarray):
    """
    A subclass of np.ndarray that has a ``meta`` and a ``bandInfo`` attribute.
    This defines the class that the LightField and SpectralImage class
    are derived from.

    A SpectralArray can be a 1D spectrum, 3D multi- (or hyper-) spectral image
    as well as regular RGB image or a 4D monochromatic or 5D color light field.
    Basically, all these concrete images are wavelength dependent arrays
    for which this base class provides the common architecture and methods.
    The shape of the arrays is (..., num_ch), i.e. (x, y, num_ch) for
    a "regular" or multispectral image or (u, v, s, t, num_ch) for a light
    field. That is, we are throughout using a "Color channel last" ordering.

    This class only allows dtypes np.float64, np.float32 and np.float16
    which are normalized to a range (0, 1).
    Float input must be in range (0, 1) already.
    """

    def __new__(cls,
                array: ndarray,
                dtype: Optional[Type[Union[np.float64, np.float32, np.float16]]] = None,
                copy: bool = True,
                meta: Optional[dict] = None,
                band_info: Optional[BandInfo] = None,
                ndim_exp: Optional[int] = None):
        """Create a SpectralArray from a numpy ndarray and metadata.

        Args:
            array: Input data array as numpy ndarray.

            dtype: Target datatype. Only numpy floats allowed:
                   np.float16, np.float32, np.float64. Default: np.float32

            copy: By default, a copy of the input array is forced. If 'False',
                  will try to only use a view of the input array. This might
                  not work in some cases and also depends on the Python
                  interpreter.

            meta: Meta data dictionary.

            band_info: :class:`BandInfo` object holding the the
                       spectral band properties.

            ndim_exp: Number of dimension expected of the array. This is used
                      by inherited classes.

        Raises:
            DimensionError(ndim_exp, array.ndim)
            If number of dimensions of input array does not match the specified
            ndim_exp option.

        Returns:
            SpectralArray
        """

        # Check input types
        if not isinstance(array, np.ndarray):
            raise ValueError("SpectralArray expects a numpy array.")

        # Check number of dimension
        if ndim_exp is not None and ndim_exp != array.ndim:
            raise DimensionError(exp=ndim_exp, found=array.ndim)

        if not (meta is None or isinstance(meta, dict)):
            raise ValueError("SpectralArray expects meta data to be a dict.")

        if not (band_info is None or isinstance(band_info, BandInfo)):
            raise ValueError("SpectralArray expects BandInfo or None instance.")

        # Check data size
        if array.size == 0:
            raise ValueError(
                f"Invalid array size. Must have at least one axis.")

        # Convert dtype to fit standard, this might force a copy
        if not array.dtype == dtype and dtype is not None:
            logger.warning(f"Converting type from {array.dtype} to {dtype}.")
            logger.warning(
                f"Range will be mapped to (0, 1).")

            if not copy:
                logger.warning("Dtype conversion without copying is not "
                               "possible. Ignoring 'copy' option.")

        # Set default dtype
        if dtype is None:
            dtype = np.float32

        if dtype in [np.float16, np.float32, np.float64]:
            conv_data = convert(array, dtype=dtype, force_copy=copy)

        else:
            raise ValueError(f"The passed dtype '{dtype}' is invalid. "
                             "Only numpy float types are allowed.")

        # Get object
        ob = conv_data.view(cls)

        # Get number of color channels from last axis
        ob._numChannels = ob.shape[-1]

        # Check that BandInfo is compatible with data
        ob._bandInfo = band_info
        if ob._bandInfo is not None \
                and not ob._numChannels == ob._bandInfo.num_channels:
            raise ValueError(
                f"The numbers of channels of the band info object "
                f"and the numbers of channels of the given data "
                f"does not match. Found "
                f"{ob._bandInfo.num_channels} and "
                f"{ob._numChannels} respectively.")

        # Set metadata
        ob._meta = meta if meta is not None else {}

        return ob

    def __array_finalize__(self, ob):
        """ Contain num_channels, meta and BandInfo on array operations.

        It is not checked if the array leaves the (0, 1) range as it
        would be too heavy on perfomance.
        """

        if isinstance(ob, SpectralArray):
            self._numChannels = ob.num_channels
            self._meta = ob.meta
            self._bandInfo = ob.band_info

        else:
            self._numChannels = ob.shape[-1]
            self._meta = {}
            self._bandInfo = None

        return

    def copy(self, order='C'):
        """Copy SpectralArray"""
        return SpectralArray(array=np.asarray(self).copy(order=order),
                             dtype=self.dtype, copy=False, meta=self.meta,
                             band_info=self.band_info)

    def save(self, path: Any, create_dir: bool, dtype: Union[np.uint8, np.uint16]):
        """Implementation of save method depends on array shape.

        Is overwritten in the :class:`SpectralImage`, resp.
        :class:`LightField` class.

        """
        pass

    def save_rgb(self, path: Any):
        """Implementation of save method depends on array shape.

        Is overwritten in the :class:`SpectralImage`, resp.
        :class:`LightField` class.

        """
        pass

    @property
    def num_channels(self):
        return self._numChannels

    @property
    def meta(self):
        return self._meta

    @property
    def band_info(self):
        return self._bandInfo

    @staticmethod
    def read_envi_header(file):
        """Read ENVI header file.

        This function is mostly analogous to the ENVI I/O of the
        SpectralPython library by Thomas Boggs releases under GNU GPGv2.
        See: https://github.com/spectralpython/spectral/blob/master/spectral/io/envi.py

        Args:
            file: Input ENVI header file (.hdr).

        Returns:
            Dictionary of header file values.
        """
        f = open(file, 'r')

        try:
            is_envi = f.readline().strip().startswith('ENVI')
        except UnicodeDecodeError:
            f.close()
            raise IOError('File is not an ENVI header file.')
        else:
            if not is_envi:
                f.close()
                raise IOError('File is not an ENVI header file.')

        lines = f.readlines()
        f.close()

        d = dict()

        try:
            while lines:
                line = lines.pop(0)
                if line.find('=') == -1:
                    continue
                if line[0] == ';':
                    continue

                (key, sep, val) = line.partition('=')
                key = key.strip().lower()
                val = val.strip()
                if val and val[0] == '{':
                    str = val.strip()
                    while str[-1] != '}':
                        line = lines.pop(0)
                        if line[0] == ';':
                            continue

                        str += '\n' + line.strip()

                    if key == 'description':
                        d[key] = str.strip('{}').strip()

                    else:
                        vals = str[1:-1].split(',')
                        for j in range(len(vals)):
                            vals[j] = vals[j].strip()
                        d[key] = vals
                else:
                    d[key] = val

            return d
        except:
            raise IOError("Parsing of ENVI header failed.")

    def normalize(self, by="peak"):
        """Normalize a SpectralArray in place.

        Args:
            by : Used normalization method. Available methods are:\n
                * 'area': normalize to sum one
                * 'peak': normalize to maximum value
                * 'l2': normalize by L2 norm of the array

        See Also:
             Based on :func:`plenpy.utilities.images.normalize_in_place()`.
        """
        return images.normalize_in_place(self, by=by)

    def get_rgb(self,
                cmf: str = "CIE_1931_1NM",
                illuminant: str = "CIE_D65") -> ndarray:
        """Calculate RGB representation of the SpectralArray

        For example, RGB values of a spectrum, a multispectral image or
        a multispectral light field.

        Args:
            cmf: Name of the color matching function.
                See :func:`plenpy.utilities.colors.get_avail_cmfs()` for a
                list of available color matching functions.
                Default: CIE 1931 standard observer sampled at 1nm.

            illuminant: Name of the illuminant. See
                :func:`plenpy.utilities.colors.get_avail_illuminants()` for a
                list of available illuminants.
                Default: CIE D65 midday light illuminant.

        Returns:
            RGB data of shape (..., 3), where the shape in the non-color axes
            remains the same.
        """

        if self.num_channels == 1:
            tmp = self.copy()
            return np.squeeze(np.stack((tmp, tmp, tmp), axis=-1))

        elif self.num_channels == 2:
            pass

        elif self.num_channels == 3:
            return self.copy()

        # Get size of one color channel
        non_spectral_size = self[..., 0].size

        # Get shape of one color channel
        shape_basis = self[..., 0].shape

        # Get shape of output RGB
        rgb_shape = shape_basis + (3, )

        # Extract wavelengths from BandInfo:
        if self.band_info is not None and self.band_info.centers is not None:
            wavelength_list = self.band_info.centers

        else:
            logger.warning(f"No wavelength centers specified. "
                           f"Using standard VIS range.")

            wavelength_list = np.linspace(430, 650, self.num_channels)

        # Get color converter instances
        # Wavelength converter, calculate RGB values of wavelengths
        wl_converter = colors.WavelengthConverter(cmf=cmf,
                                                  illuminant=illuminant)
        wavelength_rgb = wl_converter.to_rgb(wavelength_list)

        # Spectrum converter
        sp_converter = colors.SpectrumConverter(
            wavelength_list, cmf=cmf, illuminant=illuminant)

        # Reshape to get one spectrum per column
        # Squeeze necessary for 1D spectrum
        spectra = np.squeeze(
            np.reshape(self, (non_spectral_size, self.num_channels)))
        rgb = sp_converter.to_rgb(spectra)
        rgb_data = np.reshape(rgb, rgb_shape)

        return rgb_data

    def get_grey(self, weighted: bool = True) -> ndarray:
        """Calculate monochromatic from SpectralArray values.
        Uses to_rgb() conversion and then performs an RGB to grey conversion.

        Args:
            weighted: Flag indicating whether to use weighted conversion
                      according to CIE 1931 standard (default) or non weighted.

        Returns:
            Monochromatic data of shape (..., 1), where the shape in the
            non-color axes remains the same.
        """

        def rgb_to_grey(input: SpectralArray, weighted=weighted):
            """Calculate grey from RGB values."""

            if weighted is True:
                rgb = np.asarray([0.2126, 0.7152, 0.0722])

            else:
                rgb = np.asarray([1.0/3, 1.0/3, 1.0/3])

            return (input @ rgb)[..., np.newaxis]

        # Check color channels
        if self.num_channels == 1:
            return self.copy()

        elif self.num_channels == 2:
            pass

        elif self.num_channels == 3:
            return rgb_to_grey(self)

        elif self.num_channels > 3:
            return rgb_to_grey(self.get_rgb())

    # Add synonym
    get_gray = get_grey

    def _get_filters_ideal(self, num_filters) -> ndarray:
        """Calculate ideal bandpass with the given color channels.

        The filters are spread out equidistantly across the spectral range
        of the SpectralArray. The filters are normalized to a peak of one.

        Parameters:
            num_filters: Number of returned spectral filters.

        Returns:
            A 2D numpy array containing the filters.
            The first axis of the array corresponds to the filter index.
            The filters themselves are defined along the second array axis.
        """
        filter_width = self._numChannels / num_filters

        # First axis is index of the different filter number
        # Second axis is actual filter values
        filter = np.zeros((num_filters, self._numChannels), dtype=np.float64)

        spectral_range_arr = np.arange(0, self._numChannels)

        # Iterate over the filters
        for i in range(0, num_filters):
            mask = np.where(
                np.logical_and(
                    i*filter_width <= spectral_range_arr,
                    spectral_range_arr < (i + 1)*filter_width))

            filter[i, mask] = 1.0

        return filter

    def _get_filters_gauss(self, num_filters, fwhm=5) -> ndarray:
        """Calculate gaussian bandpass with the given color channels.

        The filters are spread out equidistantly across the spectral range
        of the SpectralArray. The filters are normalized to a peak of one.

        Parameters:
            num_filters: Number of returned spectral filters.

            fwhm: Full width half maximum of the filters in units of
                channels indices.

        Returns:
            A 2D numpy array containing the filters.
            The first axis of the array corresponds to the filter index.
            The filters themselves are defined along the second array axis.
        """
        # Calculate the filter width for the ideal filters
        # This is used to calculate the central wavelengths
        filter_width = self._numChannels / num_filters

        # Calculate the central wavelengths
        mu = filter_width * (0.5 + (np.arange(0, num_filters)))

        sigma = fwhm / 2.35482
        sigma_sq_inv = (1 / sigma) ** 2

        spectral_range_arr = np.arange(0, self._numChannels)
        # First axis is index of the different filter number
        # Second axis is actual filter values
        filter = np.exp(-0.5 * sigma_sq_inv * (
                spectral_range_arr - mu[np.newaxis, :].T) ** 2)

        return filter

    def get_subband(self, start: int, stop: int) -> 'SpectralArray':
        """Get a spectral subband of the SpectralArray.

        The subband is specified by a start index and stop index which is not
        included as is usual for Python array indexing x[..., start:stop].

        Args:
            start: Index of the start wavelength.

            stop: Index of the stop wavelength (not included).

        Returns:
            SpectralArray corresponding to a subband of the input SpectralArray.

        """
        # Get subband
        res = np.asarray(self)[..., start:stop]

        # Get BandInfo object
        band_info = None
        if self.band_info is not None:
            band_info = self.band_info.get_subband(start, stop)

        return SpectralArray(res, self.dtype, copy=False, meta=self.meta, band_info=band_info)

    def get_rescaled(self, scale: Union[float, Tuple[float]], **kwargs) -> 'SpectralArray':
        """Rescale the array spatially after applying an anti-aliasing filter.

        This has to be implemented by the derived class.


        Args:
            scale: Scale factors. Separate scale factors per axis can be defined.
                   If one value is given, the same factor is applied along all axes.
            **kwargs: Options passed to :func:`scipy.signal.decimate`.

        Returns:
            The spatially downsampled SpectralArray.

        """
        raise NotImplementedError("This has to be implemented by the derived class.")

    def get_resized(self, output_shape: Union[Tuple[int], ndarray], **kwargs) -> 'SpectralArray':
        """Resize the array spatially to a defined output shape.

        This has to be implemented by the derived class.


        Args:
            output_shape: Shape of the resized array.
            **kwargs: Options passed to :func:`scipy.signal.decimate`.

        Returns:
            The spatially resized SpectralArray.

        """
        raise NotImplementedError("This has to be implemented by the derived class.")

    def get_decimated_spectrum(self, q: int, **kwargs) -> 'SpectralArray':
        """Downsample the spectrum after applying an anti-aliasing filter
        in the spectral domain.

        Note that for large SpectralArrays, this consumes a lot of RAM.
        In this case, patching in the spatial domain can be applied.

        The BandInfo and metadata are updated accordingly.
        Note, that it is implied, that the spectrum is sampled in an
        equidistant fashion. If the spectrum is irregularly sampled, the
        downsampling will yield incorrect results.

        The routine mainly uses :func:`scipy.signal.decimate` for the
        downsampling.


        Args:
            q: Downsampling factor.
            **kwargs: Options passed to :func:`scipy.signal.decimate`.

        Returns:
            The spectrally downsampled SpectralArray.

        """
        # Downsample by q
        res = decimate(np.asarray(self), q=q, axis=-1, **kwargs)

        # Get updated BandInfo object
        band_info = None
        if self.band_info is not None:
            band_info = self.band_info.get_downsampled(num=res.shape[-1])

        return SpectralArray(res, self.dtype, copy=False, meta=self.meta, band_info=band_info)

    def get_resampled_spectrum(self, num: int, **kwargs) -> 'SpectralArray':
        """Resample the spectrum using FFT.

        Note that for large SpectralArrays, this consumes a lot of RAM.
        In this case, patching in the spatial domain can be applied.

        The BandInfo and metadata are updated accordingly.
        Note, that it is implied, that the spectrum is sampled in an
        equidistant fashion. If the spectrum is irregularly sampled, the
        downsampling will yield incorrect results.

        The routine mainly uses :func:`scipy.signal.resample` for the
        downsampling.


        Args:
            num: Number of channels in the output SpectralArray.
            **kwargs: Options passed to :func:`scipy.signal.decimate`.

        Returns:
            The downsampled SpectralArray.

        """

        # Downsample to num channels
        res = resample(np.asarray(self), num=num, axis=-1, **kwargs)

        # Get updated BandInfo object
        band_info = None
        if self.band_info is not None:
            band_info = self.band_info.get_downsampled(num=res.shape[-1])

        return SpectralArray(res, self.dtype, copy=False, meta=self.meta, band_info=band_info)

    def get_msi(self,
                num_ch: int,
                method: str = 'gauss',
                normalize: str = 'area',
                use_compl: bool = False) -> 'SpectralArray':
        """Get multispectral array from hyperspectral array using
        a Gaussian or ideal filter. Only the spectral axis will be altered.

        Args:
            num_ch: Number of spectral channels of downsampled image.

            method: Method used for downsampling.
                If ``gauss`` is specified, use gaussian filters,
                if ``ideal`` is specified, use ideal, rectangular filters.

            normalize: Method used to normalize the filters. By default, the
                filters are normlized by area, i.e. each filter sums to one.
                This guarantees that the output image is still bounded by 1.
                If ``peak`` is specified, the filters are normlized to peak 1.
                This results in the output image not being bound by 1. The
                output has to be down-amplified afterwards. But it comes in
                handy when looking at actual filters to be applied to a camera.

            use_compl: Boolean flag indicating whether to use complementary
                filters.

        Returns: SpectralArray
            Downsampled image.
        """

        if num_ch > self.num_channels:
            raise ValueError(f"Number of downsampled channels is larger than "
                             f"original. Please specify fewer channels.")

        if num_ch == self.num_channels:
            logger.warning(f"Number of channels of original and downsampled "
                           f"image are equal. Returning copy.")
            return self.copy()

        if method == 'gauss':
            filters = self._get_filters_gauss(num_filters=num_ch)
        elif method == 'ideal':
            filters = self._get_filters_ideal(num_filters=num_ch)
        else:
            raise ValueError(f"Unknown method {method}.")

        if use_compl:
            filters = 1 - filters

        if normalize == 'area':
            filters = filters / np.sum(filters, axis=1)[..., np.newaxis]

        elif normalize == 'peak':
            # Filter already normalized by peak
            pass

        else:
            raise ValueError(f"Unknown normalization {normalize}.")

        # TODO: BandInfo could be adapted accordingly
        # TODO: What about metadata?

        # Calculate weighted sum along spectral axis
        return SpectralArray(np.dot(self, filters.T))


def read_mat(path: Path, key: str) -> ndarray:
    """Read array data from a Matlab mat file from a specific key.

    This function wraps both ``scipy.io.loadmat()`` and ``hdf5`` to cover
    all possible file versions.

    Args:
        path: Path to mat file.
        key:  Key name of the data in ``.mat`` file. If only one variable
              is contained, option is ignored and key is found automatically.
              If more than one key is contained, the ``key`` option must be
                 specified.

    Returns:
        Data.
    """
    keys = []
    try:
        f = scipy.io.loadmat(str(path.absolute()))
        for tmp_key, value in f.items():
            if not re.findall('__.*__', tmp_key) == [tmp_key]:
                keys.append(tmp_key)
    except NotImplementedError:
        # File is HDF5 mat file
        f = h5py.File(path.absolute(), 'r+')
        for tmp_key in f.keys():
            if not re.findall('__.*__', tmp_key) == [tmp_key]:
                keys.append(tmp_key)

    if len(keys) == 0:
        raise ValueError("Did not find a data key in Matlab file.")

    if not len(keys) == 1:
        if key is None:
            raise ValueError(f"Found more than one key in Matlab file. "
                             f"Keys found: {keys}. Please specify a single"
                             f" key using the 'key' option.")
        else:
            if key in keys:
                keys = [key]
            else:
                raise KeyError(f"No key '{key}' in keys '{keys}'.")

    logger.info(f"Found key '{keys[0]}' in Matlab file with data of shape "
                f"{f[keys[0]].shape}.")
    # Get data from variable
    return np.asarray(f[keys[0]])
