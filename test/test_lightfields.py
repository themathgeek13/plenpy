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


"""Tests for plenpy.lightfields module.

"""

import imageio
import numpy as np
from pytest import raises

from plenpy import testing
import plenpy.logg

from plenpy.lightfields import LightField, BandInfo
from plenpy.spectral import SpectralImage
from plenpy.utilities.core import DimensionError

# Logging settings
logger = plenpy.logg.get_logger()
plenpy.logg.set_level("warning")

# SET LIGHTFIELD TEST DATA
# Set LF parameters
u_max = 9
v_max = 6
s_max = 12
t_max = 13

test_lf_data = np.random.rand(u_max, v_max, s_max, t_max, 1).astype(np.float32)
test_lf_data_color = np.random.rand(u_max, v_max, s_max, t_max, 3).astype(np.float32)
test_lf_data_spectral = np.random.rand(u_max, v_max, s_max, t_max, 10).astype(np.float32)
test_lf_data_color_symmetric = np.random.rand(u_max, u_max, s_max, s_max, 3).astype(np.float32)

# Get 2D rep from Light Field by copying subapertures
a = u_max * s_max
b = v_max * t_max

test_lf_data_2D_sai = np.zeros((a, b, 3)).astype(test_lf_data_color.dtype)
for u in range(0, u_max):
    for v in range(0, v_max):
        test_lf_data_2D_sai[u * s_max:(u + 1) * s_max, v * t_max:(v + 1) * t_max] = \
            test_lf_data_color[u, v, :, :]

test_lf_data_2D_sai_symmetric = np.zeros((a, a, 3)).astype(test_lf_data_color_symmetric.dtype)
for u in range(0, u_max):
    for v in range(0, u_max):
        test_lf_data_2D_sai_symmetric[u * s_max:(u + 1) * s_max, v * s_max:(v + 1) * s_max] = \
            test_lf_data_color_symmetric[u, v, :, :]

test_lf_data_2D_mli = np.zeros((a, b, 3)).astype(test_lf_data_color.dtype)
for s in range(0, s_max):
    for t in range(0, t_max):
        test_lf_data_2D_mli[s * u_max:(s + 1) * u_max,
        t * v_max:(t + 1) * v_max] = \
            test_lf_data_color[:, :, s, t]

test_lf_data_2D_mli_symmetric = np.zeros((a, a, 3)).astype(test_lf_data_color_symmetric.dtype)
for s in range(0, s_max):
    for t in range(0, s_max):
        test_lf_data_2D_mli_symmetric[s * u_max:(s + 1) * u_max,
        t * u_max:(t + 1) * u_max] = \
            test_lf_data_color_symmetric[:, :, s, t]


def test_dim_conversion_sai():
    lf = LightField(test_lf_data_color)
    data = lf.get_2d(method='sai')
    assert np.array_equal(data, test_lf_data_2D_sai)

    data = LightField._2d_to_4d(test_lf_data_2D_sai, s_max=s_max, t_max=t_max, method='sai')
    assert np.array_equal(data, test_lf_data_color)

    lf = LightField(test_lf_data_color_symmetric)
    data = lf.get_2d(method='sai')
    assert np.array_equal(data, test_lf_data_2D_sai_symmetric)

    data = LightField._2d_to_4d(test_lf_data_2D_sai_symmetric, s_max=s_max, t_max=s_max, method='sai')
    assert np.array_equal(data, test_lf_data_color_symmetric)
    return


def test_dim_conversion_mli():
    lf = LightField(test_lf_data_color)
    data = lf.get_2d(method='mli')
    assert np.array_equal(data, test_lf_data_2D_mli)

    data = LightField._2d_to_4d(test_lf_data_2D_mli, s_max=s_max, t_max=t_max, method='mli')
    assert np.array_equal(data, test_lf_data_color)

    lf = LightField(test_lf_data_color_symmetric)
    data = lf.get_2d(method='mli')
    assert np.array_equal(data, test_lf_data_2D_mli_symmetric)

    data = LightField._2d_to_4d(test_lf_data_2D_mli_symmetric, s_max=s_max, t_max=s_max, method='mli')
    assert np.array_equal(data, test_lf_data_color_symmetric)
    return


def test_2d_and_4d_bijectivitiy():

    lf = LightField(test_lf_data_color)
    lf_ap = lf.apply_aperture('circ')

    # Test RECT grid
    for method in ['MLI', 'SAI']:
        data = lf.get_2d(method=method, grid='rect')
        lf_rec = LightField._2d_to_4d(data, s_max=s_max, t_max=t_max,
                                      method=method, grid='rect')
        assert np.array_equal(lf, lf_rec)

        data = lf.get_2d(method=method, aperture='circ', grid='rect')
        lf_rec = LightField._2d_to_4d(data, s_max=s_max, t_max=t_max,
                                      method=method, grid='rect')
        assert np.array_equal(lf_ap, lf_rec)

    # For hex grid, use circular aperture
    data = lf.get_2d(method='MLI', aperture='circ', grid='hex')
    lf_rec = LightField(LightField._2d_to_4d(data, s_max=s_max, t_max=t_max,
                                             method='MLI', grid='hex'))
    # Just test for fail, as reconstruction will be lossy due to interpolation...

    return


def test_lightfield_init():

    test_lf = LightField(test_lf_data)
    test_lf_color = LightField(test_lf_data_color)
    test_lf_spectral = LightField(test_lf_data_spectral)

    u, v, s, t, ch = test_lf.shape

    assert ch == 1
    assert u == u_max
    assert v == v_max
    assert s == s_max
    assert t == t_max
    assert test_lf_color.num_channels == 3
    assert test_lf_spectral.num_channels == 10

    assert np.array_equal(test_lf, test_lf_data)
    assert np.array_equal(test_lf_color, test_lf_data_color)
    assert np.array_equal(test_lf_spectral, test_lf_data_spectral)

    # Test correct expted dimension
    with raises(DimensionError) as cm:
        tmp = LightField(np.random.rand(4, 4, 3, 3), dtype=np.float32, copy=True)

    assert f"Expected 5D input. Found 4D." == str(cm.value)

    return


def test_constructor_from_img_data():
    img_data_lf = LightField.from_img_data(img=test_lf_data_2D_sai,
                                           s_max=s_max,
                                           t_max=t_max)

    assert np.array_equal(img_data_lf, test_lf_data_color)

    # Test symmetric light field
    img_data_lf = LightField.from_img_data(img=test_lf_data_2D_sai_symmetric,
                                           s_max=s_max)

    assert np.array_equal(img_data_lf, test_lf_data_color_symmetric)

    return


def test_constructor_from_file():
    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Save test image
        file = path / "test_lf.png"
        imageio.imsave(uri=file, im=test_lf_data_2D_sai)

        # Read from saved file
        path_lf = LightField.from_file(file, s_max=s_max, t_max=t_max)

        assert np.allclose(path_lf, test_lf_data_color, atol=0.01)

    finally:
        # Cleanup folder
        testing.remove_tmp_folder(id)
    return


def test_constructor_from_file_mask():
    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Save 1-channel test image
        file = path / "test_lf.png"
        mask_file = path / "test_mask_array.npy"

        imageio.imsave(uri=file, im=test_lf_data_2D_sai[..., 0])

        # Test layer wise 1 mask
        # Create mask array with 10 color channels
        for ch in range(0, 10):
            mask = np.zeros((s_max, t_max, 10))
            mask[..., ch] = 1.0

            np.save(mask_file, mask)

            # Create test masked lightfield
            path_lf= LightField.from_file_mask(file, mask_path=mask_file)

            assert np.allclose(path_lf[:, :, :, :, ch],
                               test_lf_data_color[..., 0],
                               atol=0.01)

        # Test all zero  mask
        mask = np.zeros((s_max, t_max, 10))
        np.save(mask_file, mask)
        path_lf = LightField.from_file_mask(file, mask_path=mask_file)

        # Elements, where mask is zero are set to nan
        assert np.all(np.isnan(path_lf))

        # Test pixelwise 1 mask
        for ch in range(0, 10):
            for s in range(0, s_max):
                for t in range(0, t_max):
                    mask = np.zeros((s_max, t_max, 10))
                    mask[s, t, ch] = 1.0
                    np.save(mask_file, mask)

                    # Create test masked lightfield
                    path_lf= LightField.from_file_mask(file, mask_path=mask_file)

                    # create ref data
                    ref_data = np.nan*np.zeros((u_max, v_max, s_max, t_max, 10))
                    # only entry where mask is 1
                    ref_data[:, :, s, t, ch] = test_lf_data_color[:, :, s, t, 0]

                    np.place(path_lf, np.isnan(path_lf), 0)
                    np.place(ref_data, np.isnan(ref_data), 0)
                    assert np.allclose(path_lf, ref_data, atol=0.01)

        # Test inconsistent mask and img shape
        mask = np.zeros((s_max-1, t_max+1, 10))
        np.save(mask_file, mask)

        with raises(ValueError) as cm:
            LightField.from_file_mask(file, mask_path=mask_file)

        assert "The mask and image shapes are incompatible." == str(cm.value)

    finally:
        testing.remove_tmp_folder(id)

    return


def test_constructor_from_file_collection():
    # Get a tmp folder
    folder, id = testing.get_tmp_folder()

    try:
        for u in range(0, u_max):
            for v in range(0, v_max):
                # Set blockwise constant entries and save
                filename = folder / f"lf_image_{u}_{v}.png"
                imageio.imsave(uri=filename, im=test_lf_data_color[u, v])

        path_collection_lf = LightField.from_file_collection(
            path=folder, u_max=u_max, v_max=v_max)

        assert np.allclose(path_collection_lf, test_lf_data_color, atol=0.01)

    finally:
        testing.remove_tmp_folder(id)

    # Get a tmp folder
    folder, id = testing.get_tmp_folder()

    try:

        for u in range(0, u_max):
            for v in range(0, v_max):
                # Set blockwise constant entries
                filename = folder / f"lf_image_{u}_{v}.png"
                imageio.imsave(uri=filename, im=test_lf_data[u, v])

        path_collection_lf = LightField.from_file_collection(
            path=folder, u_max=u_max, v_max=v_max)

        assert np.allclose(path_collection_lf, test_lf_data, atol=0.01)

    finally:
        testing.remove_tmp_folder(id)

    # Get a tmp folder
    folder, id = testing.get_tmp_folder()

    try:
        for u in range(0, u_max):
            for v in range(0, u_max):
                # Set blockwise constant entries
                filename = folder / f"lf_image_{u}_{v}.png"
                imageio.imsave(uri=filename, im=test_lf_data_color_symmetric[u, v])

        path_collection_lf = LightField.from_file_collection(
            path=folder, u_max=u_max)

        assert np.allclose(path_collection_lf, test_lf_data_color_symmetric, atol=0.01)

    finally:
        testing.remove_tmp_folder(id)

    # Get a tmp folder
    folder, id = testing.get_tmp_folder()

    try:

        # Test invalid number and shapes
        # Save test images

        for u in range(0, u_max):
            for v in range(0, v_max):
                # Set blockwise constant entries
                filename = folder / f"lf_image_{u}_{v}.png"
                imageio.imsave(uri=filename, im=test_lf_data_color[u, v])

        # Delete one image
        (folder / "lf_image_1_2.png").unlink()

        with raises(ValueError) as cm:
            path_collection_lf = LightField.from_file_collection(
                path=folder, u_max=u_max, v_max=v_max)

        assert (f"Found {u_max * v_max - 1} images. "
                f"Expected {u_max * v_max}. Please check that all "
                f"images in the folder are readable.") == str(cm.value)

        # Create missing image, but with wrong shape
        imageio.imsave(uri=(folder / "lf_image_1_2.png"),
                       im=np.ones((4, 4, 3), dtype=np.uint8))

        with raises(ValueError) as cm:
            path_collection_lf = LightField.from_file_collection(
                path=folder, u_max=u_max, v_max=v_max)

        assert f"The images are not all of the same shape." == str(cm.value)

    finally:
        # Cleanup
        testing.remove_tmp_folder(id)

    return


def test_save():
    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Create example image
        u, v, s, t, ch = 5, 4, 64, 32, 30
        arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

        tmp = LightField(arr, dtype=np.float32, copy=False)

        file = path / 'test.png'
        for d in [np.uint8, np.uint16]:
            tmp.save(path=file, dtype=d)

            # Load image
            # First load Spectral Image
            # Load image
            hsi_loaded = SpectralImage.from_file_collection(path=path,
                                                            dtype=np.float32)

            loaded = LightField.from_img_data(hsi_loaded, s_max=s, t_max=t)

            # Caution: lossy dtype conversion
            assert np.allclose(arr,  loaded, atol=0.01)

    finally:
        # Cleanup temporary folder
        testing.remove_tmp_folder(id)

    return


def test_save_rgb():
    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Create example image
        u, v, s, t, ch = 5, 4, 64, 32, 30
        arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

        tmp = LightField(arr, dtype=np.float32, copy=False)
        tmp_rgb = tmp.get_rgb()

        rgb_filename = path / "test_rgb_img.png"
        tmp.save_rgb(rgb_filename)

        # Load image
        loaded = LightField.from_file(path=rgb_filename,
                                      s_max=s, t_max=t,
                                      dtype=np.float32)

        # Caution: lossy dtype conversion
        assert np.allclose(tmp_rgb, loaded, atol=0.01)

    finally:
        # Cleanup temporary folder
        testing.remove_tmp_folder(id)

    return


def test_get_colorcoded_copy_parsing():
    test_lf = LightField(test_lf_data_color)

    # Test error handling
    with raises(ValueError) as cm:
        tmp = test_lf.get_colorcoded_copy(method="Unknown")
    assert "Unknown method: 'Unknown'." == str(cm.value)

    with raises(ValueError) as cm:
        tmp = test_lf.get_colorcoded_copy(method=4.0)
    assert "Unknown method type: <class 'float'>." == str(cm.value)

    mask = np.ones((33, 1, 2))
    with raises(ValueError) as cm:
        tmp = test_lf.get_colorcoded_copy(method=mask)
    assert f"The given mask is of incompatibel shape.\n" \
        f"Given: {(33, 1, 2)}.\n" \
        f"Expected: {(s_max, t_max, 3)}." == str(cm.value)

    return


def test_get_colorcoded_copy():

    # Light field with three color channels
    test_lf = LightField(test_lf_data_color)

    # Test layer wise 1 mask
    for ch in range(0, 3):
        mask = np.ones((s_max, t_max, 3))
        arr = np.arange(0, 3)
        arr = np.delete(arr, ch)
        mask[..., arr] = 0

        coded_lf = test_lf.get_colorcoded_copy(method=mask, cval=np.nan)
        assert np.array_equal(coded_lf[..., ch],
                              test_lf[..., ch])

        assert np.all(np.isnan(coded_lf[..., arr]))

        coded_lf_cval_z = test_lf.get_colorcoded_copy(method=mask, cval=0)
        assert np.array_equal(coded_lf_cval_z[..., ch],
                              test_lf[..., ch])

        assert np.all(coded_lf_cval_z[..., arr] == 0)

        coded_lf_cval_n = test_lf.get_colorcoded_copy(method=mask, cval=0.9)
        assert np.array_equal(coded_lf_cval_n[..., ch],
                              test_lf[..., ch])

        assert np.all(coded_lf_cval_n[..., arr] == 0.9)

        # coded_lf = test_lf.get_colorcoded_copy(method=mask, cval=0)
        # assert np.array_equal(coded_lf[..., ch],
        #                       test_lf[..., ch])
        #
        # assert np.all(coded_lf[..., arr] == 0.0)

    # Test all zero  mask
    mask = np.zeros((s_max, t_max, 3))
    coded_lf = test_lf.get_colorcoded_copy(method=mask, cval=np.nan)
    assert np.all(np.isnan(coded_lf))

    # Test random mask
    coded_lf, mask = test_lf.get_colorcoded_copy(method="random",
                                                 cval=np.nan,
                                                 return_mask=True)

    for u in range(0, u_max):
        for v in range(0, v_max):
            assert np.array_equal(coded_lf[u, v][mask == 1], test_lf[u, v][mask == 1])
            assert np.all(np.isnan(coded_lf[u, v][np.isnan(mask)]))

    # Test regular mask

    # Fails for non square number of channels
    with raises(ValueError) as cm:
        coded_lf = test_lf.get_colorcoded_copy(method="regular", cval=np.nan)
    assert f"Regular mask only works with square number of color channels. " \
               f"Found 3." == str(cm.value)

    # For square number of channels:
    u, v, s, t, ch = 5, 4, 64, 32, 16
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)
    test_lf = LightField(arr)
    coded_lf, mask = test_lf.get_colorcoded_copy(method="regular",
                                                 cval=np.nan,
                                                 return_mask=True)

    for u in range(0, u):
        for v in range(0, v):
            assert np.array_equal(coded_lf[u, v][mask == 1],
                                  test_lf[u, v][mask == 1])
            assert np.all(np.isnan(coded_lf[u, v][np.isnan(mask)]))

    return


def test_subband():
    # Get a temporary folder
    path, id = testing.get_tmp_folder()

    try:
        # Create example image
        u, v, s, t, ch = 5, 4, 64, 32, 30
        arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

        band_info = BandInfo.from_equidistant(ch, 400, 700)
        meta = dict(metadata="test")
        tmp = LightField(arr, meta=meta, band_info=band_info, dtype=np.float32, copy=False)

        tmp_subband = tmp.get_subband(10, 20)

        assert isinstance(tmp_subband, LightField)
        assert tmp_subband.num_channels == 10
        assert meta == tmp_subband.meta
        assert np.array_equal(band_info.centers[10:20], tmp_subband.band_info.centers)

    finally:
        # Cleanup temporary folder
        testing.remove_tmp_folder(id)

    return


def test_rescale():
    # Create example image
    u, v, s, t, ch = 5, 7, 32, 64, 3
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    tmp = LightField(arr, dtype=np.float32, copy=False)

    scale = 0.5
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 16, 32, 3)

    scale = 2
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 128, 3)

    scale = (2, 1)
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 64, 3)

    scale = (2, 0.5)
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 32, 3)

    # Create example multispectral light field
    u, v, s, t, ch = 5, 7, 32, 64, 13
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    tmp = LightField(arr, dtype=np.float32, copy=False)

    scale = 0.5
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 16, 32, 13)

    scale = 2
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 128, 13)

    scale = (2, 1)
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 64, 13)

    scale = (2, 0.5)
    tmp_r = tmp.get_rescaled(scale)
    assert tmp_r.shape == (5, 7, 64, 32, 13)

    return


def test_resize():
    # Create example image
    u, v, s, t, ch = 5, 7, 32, 64, 3
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    tmp = LightField(arr, dtype=np.float32, copy=False)

    output_shape = (5, 7, 16, 16)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (5, 7, 16, 16, 3)

    output_shape = (4, 4, 32, 64)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (4, 4, 32, 64, 3)

    output_shape = (3, 5, 16, 32)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (3, 5, 16, 32, 3)

    # Create example multispectral light field
    u, v, s, t, ch = 5, 7, 32, 64, 13
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    tmp = LightField(arr, dtype=np.float32, copy=False)

    output_shape = (5, 7, 16, 16)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (5, 7, 16, 16, 13)

    output_shape = (4, 4, 32, 64)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (4, 4, 32, 64, 13)

    output_shape = (3, 5, 16, 32)
    tmp_r = tmp.get_resized(output_shape)
    assert tmp_r.shape == (3, 5, 16, 32, 13)

    return


def test_decimated():

    # Create example image
    u, v, s, t, ch = 5, 4, 64, 32, 30
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    band_info = BandInfo.from_equidistant(ch, 400, 700)
    meta = dict(metadata="test")
    tmp = LightField(arr, meta=meta, band_info=band_info, dtype=np.float32, copy=False)
    band_info_down = BandInfo.from_equidistant(10, 400, 700)

    tmp_subband = tmp.get_decimated_spectrum(3)

    assert isinstance(tmp_subband, LightField)
    assert tmp_subband.num_channels == 10
    assert meta == tmp_subband.meta
    assert band_info_down == tmp_subband.band_info

    return


def test_resampled():

    # Create example image
    u, v, s, t, ch = 5, 4, 64, 32, 30
    arr = np.random.rand(u, v, s, t, ch).astype(np.float32)

    band_info = BandInfo.from_equidistant(ch, 400, 700)
    meta = dict(metadata="test")
    tmp = LightField(arr, meta=meta, band_info=band_info, dtype=np.float32, copy=False)

    tmp_subband = tmp.get_resampled_spectrum(13)
    band_info_down = BandInfo.from_equidistant(13, 400, 700)

    assert isinstance(tmp_subband, LightField)
    assert tmp_subband.num_channels == 13
    assert meta == tmp_subband.meta
    assert band_info_down == tmp_subband.band_info

    return


def test_show_functions():
    test_lf = LightField(test_lf_data_color)
    assert test_lf.show() is None
    assert test_lf.show_subaperture(0, 0) is None
    assert test_lf.show_disparity() is None
    assert test_lf.show_2d() is None

    return


def test_get_disparity():
    test_lf = LightField(test_lf_data_color)

    # Methods for disparity estimation
    methods = ['structure_tensor', 'brute_force_2d', 'brute_force_4d']

    # According key word arguments for each method
    method_kwargs=[dict(epi_method='2d'),
                   dict(num_slopes=5),
                   dict(num_slopes=5)]

    # Possible fusion of u and v disparity
    fusions = ['average', 'weighted_average', 'max_confidence', 'tv_l1', 'no_fusion']

    for method, kwargs in zip(methods, method_kwargs):
        disp, conf = test_lf.get_disparity(method=method, vmin=-1, vmax=1, **kwargs)

        assert disp.max() <= 1
        assert disp.min() >= -1

    for fusion in fusions:
        disp, conf = test_lf.get_disparity(method=methods[0], fusion_method=fusion)

    # Check error handling
    with raises(NotImplementedError) as cm:
        test_lf.get_disparity(method='unknown')
    assert f"Method 'unknown' does not exist (yet?)" == str(cm.value)

    with raises(NotImplementedError) as cm:
        test_lf.get_disparity(method=methods[0], fusion_method='unknown')
    assert f"Fusion method 'unknown' does not exist." == str(cm.value)

    # Otherwise, no real test can be done at this point, except that it
    # should not fail...

    return


def test_show():# Monochromatic light field
    test_lf = LightField(test_lf_data)
    test_lf.show()
    test_lf.show_2d()
    test_lf.show_disparity()
    test_lf.show_subaperture(0, 0)

    # RGB light field
    test_lf = LightField(test_lf_data_color)
    test_lf.show()
    test_lf.show_2d()
    test_lf.show_disparity()
    test_lf.show_subaperture(0, 0)

    # Multispectral light field
    test_lf = LightField(test_lf_data_spectral)
    test_lf.show()
    test_lf.show_2d()
    test_lf.show_disparity()
    test_lf.show_subaperture(0, 0)

    return
