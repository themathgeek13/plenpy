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


"""Tests for plenpy.utilities.colors module.

"""
from numpy.core.multiarray import ndarray
import numpy as np
from pytest import raises, approx
from unittest.mock import patch

import plenpy.logg
from plenpy import testing
from plenpy.utilities import colors

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")


def test_get_avail_cmfs():

    cmf_list = colors.get_avail_cmfs()

    # Basic type check.
    # Check that get_cmf() can use all of the provided names.
    # Check that all kernel names are implemented

    for name in cmf_list:
        assert type(name) == str
        assert type(colors.get_cmf(cmf=name)) == ndarray

    return


def test_get_avail_illuminants():

    ill_list = colors.get_avail_illuminants()

    # Basic type check.
    # Check that get_cmf() can use all of the provided names.
    # Check that all kernel names are implemented

    for name in ill_list:
        assert type(name) == str
        assert type(colors.get_illuminant(illuminant=name)) == ndarray

    return


def test_get_parsing():

    with raises(ValueError) as cm:
        cmf = colors.get_cmf("noValidCMF")
    assert (f"The color matching function 'noValidCMF' is not one of the "
            f"available functions {colors.get_avail_cmfs()}.") == str(cm.value)

    with raises(ValueError) as cm:
        ill = colors.get_illuminant("noValidIlluminant")
    assert (f"The illuminant 'noValidIlluminant' is not one of the available "
            f"illuminants {colors.get_avail_illuminants()}.") == str(cm.value)


def test_xyz_rgb_conversion_parsing():

    NOT_VALID = np.asarray([20, 40])
    with raises(ValueError) as cm:
        rgb = colors.xyz_to_rgb(NOT_VALID)
    assert ("The passed XYZ values have to be of length 3. "
            "If You have specified multiple XYZ values, they"
            "must be stacked vertically.") == str(cm.value)
    with raises(ValueError) as cm:
        xyz = colors.rgb_to_xyz(NOT_VALID)
    assert ("The passed RGB values have to be of length 3. "
            "If You have specified multiple RGB values, they"
            "must be stacked vertically.") == str(cm.value)

    with raises(ValueError) as cm:
        rgb = colors.xyz_to_rgb(5)
    assert ("Please specify a valid XYZ value as a list or ndarray.") == str(
        cm.value)
    with raises(ValueError) as cm:
        xyz = colors.rgb_to_xyz(5)
    assert ("Please specify a valid RGB value as a list or ndarray.") == str(
        cm.value)

    return


def test_xyz_rgb_conversion():

    # Just check that multiple values are converted correctly
    xyz1 = [20, 50, 60]
    xyz2 = [10, 80, 22]
    xyz_stack = np.vstack((xyz1, xyz2))

    rgb1 = colors.xyz_to_rgb(xyz1)
    rgb2 = colors.xyz_to_rgb(xyz2)

    rgb_from_stack = colors.xyz_to_rgb(xyz_stack)
    rgb_stack = np.vstack((rgb1, rgb2))

    assert np.allclose(rgb_from_stack, rgb_stack, rtol=1E-5)

    # Do the same for RGB to XYZ
    rgb1 = [0.15, 0.5, 0.7]
    rgb2 = [0.8, 0.1, 0.22]
    rgb_stack = np.vstack((rgb1, rgb2))

    xyz1 = colors.rgb_to_xyz(rgb1)
    xyz2 = colors.rgb_to_xyz(rgb2)

    xyz_from_stack = colors.rgb_to_xyz(rgb_stack)
    xyz_stack = np.vstack((xyz1, xyz2))

    assert np.allclose(xyz_from_stack, xyz_stack, rtol=1E-5)

    return


def test_abstract_converter():
    # Create Mock Patch Abstract Base Classes
    abc_patch = patch.multiple(colors.Converter, __abstractmethods__=set())
    abc_patch.start()


    cmf_name = colors.get_avail_cmfs()[0]
    illuminant_name = colors.get_avail_illuminants()[0]

    test_converter = colors.Converter(cmf=cmf_name, illuminant=illuminant_name)

    assert cmf_name == test_converter.cmf_name
    assert illuminant_name == test_converter.illuminant_name

    assert test_converter.to_xyz() is None
    assert test_converter.to_rgb() is None

    abc_patch.stop()
    return


def test_spectrum_converter():

    # Go through all cmf-illuminator combinations
    for cmf_name in colors.get_avail_cmfs():
        for illuminant_name in colors.get_avail_illuminants():

            cmf_ref = colors.get_cmf(cmf_name)
            ill_ref = colors.get_illuminant(illuminant_name)

            # Get wavelength basis
            lambda_min = max(cmf_ref[0, 0], ill_ref[0, 0])
            lambda_max = min(cmf_ref[-1, 0], ill_ref[-1, 0])
            ref_wavelengths = np.arange(lambda_min, lambda_max + 1)

            converter = colors.SpectrumConverter(
                wavelengths=np.arange(300, 900),
                cmf=cmf_name,
                illuminant=illuminant_name)

            # Check correct cropping of passed wavelength basis
            assert np.array_equal(ref_wavelengths, converter.wavelengths)

            # Check that white spectrum is white rgb for daylight illuminant
            if illuminant_name == 'CIE_D65':
                converter = colors.SpectrumConverter(
                    wavelengths=ill_ref[:, 0],
                    cmf=cmf_name,
                    illuminant=illuminant_name)

                assert np.allclose([1, 1, 1],
                                   converter.to_rgb(ill_ref[:, 1]),
                                   atol=0.075)

        # Use the CIE 2006 reference at 0.1 nanometers to check interpolation
        cmf_ref = colors.get_cmf('CIE_2006_0_1NM')

        for cmf_name in ['CIE_2006_1NM', 'CIE_2006_5NM']:

            cmf = colors.get_cmf(cmf_name)
            converter = colors.SpectrumConverter(
                wavelengths=cmf_ref[:, 0],
                cmf=cmf_name,
                illuminant='CIE_D65')

            # Check interpolated XYZ values
            assert np.allclose(cmf_ref[:, 1], converter._x_vals, atol=0.005)
            assert np.allclose(cmf_ref[:, 2], converter._y_vals, atol=0.005)
            assert np.allclose(cmf_ref[:, 3], converter._z_vals, atol=0.005)

        # Test that multiple spectra are converted correctly
        wavelengths = np.arange(400, 701)
        spec1 = np.random.rand(len(wavelengths))
        spec2 = np.random.rand(len(wavelengths))
        spec3 = np.random.rand(len(wavelengths))
        spec_stack = np.vstack((spec1, spec2, spec3))

        converter = colors.SpectrumConverter(wavelengths=wavelengths)

        # Test XYZ stacks
        xyz1 = converter.to_xyz(spec1)
        xyz2 = converter.to_xyz(spec2)
        xyz3 = converter.to_xyz(spec3)
        xyz_stack = np.vstack((xyz1, xyz2, xyz3))
        xyz_from_stack = converter.to_xyz(spec_stack)
        assert np.allclose(xyz_stack, xyz_from_stack, rtol=1E-5)

        # Test RGB stacks
        rgb1 = converter.to_rgb(spec1)
        rgb2 = converter.to_rgb(spec2)
        rgb3 = converter.to_rgb(spec3)
        rgb_stack = np.vstack((rgb1, rgb2, rgb3))
        rgb_from_stack = converter.to_rgb(spec_stack)
        assert np.allclose(rgb_stack, rgb_from_stack, rtol=1E-5)

        # Test error handling when size of spectrum does not match
        # the wavelength basis
        wavelengths = np.arange(400, 701)
        converter = colors.SpectrumConverter(wavelengths=wavelengths)
        with raises(ValueError) as cm:
            rgb = converter.to_xyz([0.1, 0.5, 0.2])
        assert ("The specified spectrum has to have the same "
                "length as the wavelength basis that the "
                "converter was initialized with.") == str(cm.value)

    return


def test_wavelength_converter():
    # Test that multiple spectra are converted correctly

    converter = colors.WavelengthConverter()

    # Test XYZ stacks
    xyz1 = converter.to_xyz(400)
    xyz2 = converter.to_xyz(500)
    xyz3 = converter.to_xyz(600)
    xyz4 = converter.to_xyz(543)
    xyz_stack = np.vstack((xyz1, xyz2, xyz3, xyz4))
    xyz_from_stack = converter.to_xyz([400, 500, 600, 543])
    assert np.allclose(xyz_stack, xyz_from_stack, rtol=1E-5)

    rgb1 = converter.to_rgb(400)
    rgb2 = converter.to_rgb(500)
    rgb3 = converter.to_rgb(600)
    rgb4 = converter.to_xyz(543)
    rgb_stack = np.vstack((rgb1, rgb2, rgb3))
    rgb_from_stack = converter.to_rgb([400, 500, 600, 543])
    assert np.allclose(xyz_stack, xyz_from_stack, rtol=1E-5)

    return
