# Copyright (C) 2018  The Plenpy Authors
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


"""Tests for plenpy.cameras.abstract_lightfield_camera module.

"""

from pathlib import Path
import numpy as np

import plenpy.logg

from plenpy import __name__ as APPNAME
from plenpy import testing
from plenpy.cameras.lytro_illum import LytroIllum
from plenpy.lightfields import LightField

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

APP_FOLDER = testing.appdata_dir(APPNAME)

# Global test names
BASE_FOLDERPATH = Path(APP_FOLDER).absolute()
CAMERA_FOLDERPATH = BASE_FOLDERPATH / "cameras/"
LYTRO_FOLDERPATH = CAMERA_FOLDERPATH / "LytroIllum/"

FILES_LYTRO = [LYTRO_FOLDERPATH / "Calibration/cal_data.npz",
               LYTRO_FOLDERPATH / "Calibration/cal_own.npz",
               LYTRO_FOLDERPATH / "Calibration/cal_dans.npz"]

LYTRO_SMALL_FOLDERPATH = CAMERA_FOLDERPATH / "LytroIllumSmall/"

FILES_LYTRO_SMALL = [LYTRO_SMALL_FOLDERPATH / "Calibration/cal_data.npz",
                     LYTRO_SMALL_FOLDERPATH / "Calibration/cal_own.npz",
                     LYTRO_SMALL_FOLDERPATH / "Calibration/cal_dans.npz"]


def create_test_lytro_camera():
    """Download the Lytro test camera, if not already present."""
    testing.needs_internet()

    for file in ["MOD_0033"]:
        testing.get_remote_file("cameras/LytroIllum/Calibration/" + file + ".RAW")
        testing.get_remote_file("cameras/LytroIllum/Calibration/" + file + ".TXT")

    testing.get_remote_file("cameras/LytroIllum/Images/IIIT.LFR")

    return


def create_test_lytro_camera_small():
    """Download the Lytro test camera, if not already present."""
    testing.needs_internet()

    for file in ["MOD_0001", "MOD_0016", "MOD_0033"]:
        testing.get_remote_file("cameras/LytroIllumSmall/Calibration/" + file + ".bsdf")

    testing.get_remote_file("cameras/LytroIllumSmall/Images/IIIT.bsdf")

    return


def delete_test_lytro_camera():
    """ Helper function to delete temporary files.
    """
    for file in FILES_LYTRO:
        try:
            file.unlink()
        except FileNotFoundError:
            pass
    return


def delete_test_lytro_camera_small():
    """ Helper function to delete temporary files.
    """
    for file in FILES_LYTRO_SMALL:
        try:
            file.unlink()
        except FileNotFoundError:
            pass

    return


def test_lytro_illum_init():
    create_test_lytro_camera()

    try:
        test_camera = LytroIllum(LYTRO_FOLDERPATH)

        # Check member variables
        assert test_camera._microlensSize == 14
        assert test_camera._microlensRadius == 7
        assert test_camera._gridType == 'hex'
        assert test_camera._microlensFocalLength == 40e-6
        assert test_camera._calibrationDB is None
        assert test_camera._whiteImageDB is None
        assert test_camera._calDataFilename == Path('Calibration/cal_data.npz')

    finally:
        # Always clean up
        delete_test_lytro_camera()

    return


def test_lytro_illum_calibration():
    """Simple test of LytroIllum camera. Only check that no Error is raised.
    """
    create_test_lytro_camera()

    try:
        test_camera_dans = LytroIllum(LYTRO_FOLDERPATH)

        # Calibrate
        test_camera_dans.calibrate(method='dans', force=True)

        # Decode
        test_camera_dans.load_sensor_image(0)
        test_camera_dans.decode_sensor_image(0)
        lf = test_camera_dans.get_decoded_image(0)

        assert lf.shape == (13, 13, 430, 622, 3)

    finally:
        # Always clean up
        delete_test_lytro_camera()

    return


def test_lytro_camera_small():
    """Perform tests on LytroIllum camera with cropped images
    to speed up testing.
    """
    create_test_lytro_camera_small()

    try:
        test_camera_dans = LytroIllum(LYTRO_SMALL_FOLDERPATH, format='BSDF')
        test_camera_own = LytroIllum(LYTRO_SMALL_FOLDERPATH, format='BSDF')

        # Calibrate
        test_camera_dans.calibrate(filename="cal_dans.npz", method='dans', force=True)
        test_camera_own.calibrate(filename="cal_own.npz", method='own', force=True)

        # Create new cam, load data and compare
        test_camera_dans_2 = LytroIllum(LYTRO_SMALL_FOLDERPATH)
        test_camera_dans_2.calibrate(filename="cal_dans.npz", method='dans', force=False)
        test_camera_own_2 = LytroIllum(LYTRO_SMALL_FOLDERPATH)
        test_camera_own_2.calibrate(filename="cal_own.npz", method='dans', force=False)

        for i in range(test_camera_dans._calibrationDB.size):
            # Assert equality of calibration data
            for arg in ['path', 'ideal_grid_params', 'align_grid_params']:
                assert test_camera_dans._calibrationDB[i][arg] == \
                       test_camera_dans_2._calibrationDB[i][arg]
                assert test_camera_own._calibrationDB[i][arg] == \
                       test_camera_own_2._calibrationDB[i][arg]

            assert np.array_equal(test_camera_dans._calibrationDB[i]['align_transform'].params,
                                  test_camera_dans_2._calibrationDB[i]['align_transform'].params)

            assert np.array_equal(test_camera_own._calibrationDB[i]['align_transform'].params,
                                  test_camera_own_2._calibrationDB[i]['align_transform'].params)

        # Test Decoding
        del test_camera_dans_2, test_camera_own_2

        test_camera_own.load_sensor_image(0, format='BSDF')
        test_camera_own.decode_sensor_image(0)
        lf_own = test_camera_own.get_decoded_image(0)

        test_camera_dans.load_sensor_image(0, format='BSDF')
        test_camera_dans.decode_sensor_image(0)
        lf_dans = test_camera_dans.get_decoded_image(0)

        assert lf_dans.shape == (13, 13, 36, 37, 3)
        assert lf_own.shape == (13, 13, 36, 37, 3)
        assert np.allclose(lf_own, lf_dans, atol=0.5)

    finally:
        # Always clean up
        delete_test_lytro_camera_small()

    return
