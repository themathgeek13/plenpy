#!/usr/bin/env python

__author__ = "Rohan Rao"
__email__ = "rgrao@andrew.cmu.edu"
__license__ = """
    Copyright (c) 2020 Rohan Rao <rgrao@andrew.cmu.edu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from plenpy.lightfields import LightField
from plenpy.cameras.lytro_illum import LytroIllum
from pylab import *
import numpy as np

def generate_digital_zoom(inputimg, inputdepth, zoom_factor):
    x1A, y1A, __ = np.indices(inputimg.shape)
    H, W, C = inputimg.shape

    u0 = [H//2, W//2]

    x1B = zoom_factor*x1A + (1-zoom_factor)*u0[0]
    x1B = x1B.astype(np.int16).transpose(2,0,1)
    y1B = zoom_factor*y1A + (1-zoom_factor)*u0[1]
    y1B = y1B.astype(np.int16).transpose(2,0,1)

    zoomimg = np.zeros_like(inputimg)
    zoomdepth = np.zeros_like(inputdepth)
    for i in range(C):
        zoomimg[x1B[i],y1B[i],i] = inputimg[:, :, i]
    zoomdepth[x1B[0], y1B[0]] = inputdepth[:,:]

    return zoomimg, zoomdepth

class SynthesisPipeline(object):
    
    def __init__(self, *args, **kwargs):
        self.D0 = kwargs['D0'] if 'D0' in kwargs else 0.5
        self.t  = kwargs['t'] if 't' in kwargs else 0.2
        self.u0 = kwargs['u0'] if 'u0' in kwargs else None
        self.maskf = 1000

    def generate_synthesized_views(self, inputimg, inputdepth):
        x1A, y1A, __ = np.indices(inputimg.shape)
        H, W, C = inputimg.shape
        if self.u0 is None:
            self.u0 = [H//2, W//2]

        self.D1A = np.dstack([inputdepth, inputdepth, inputdepth])
        x1B = self.D1A*(self.D0-self.t)*x1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[0]/(self.D0*(self.D1A-self.t))
        x1B = x1B.astype(np.int16).transpose(2,0,1)
        y1B = self.D1A*(self.D0-self.t)*y1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[1]/(self.D0*(self.D1A-self.t))
        y1B = y1B.astype(np.int16).transpose(2,0,1)

        synthimg = np.ones_like(inputimg)*self.maskf
        synthdepth = np.ones_like(inputdepth)*self.maskf
        for i in range(C):
            synthimg[x1B[i],y1B[i],i] = inputimg[:, :, i]
        synthdepth[x1B[0], y1B[0]] = inputdepth[:, :]

        return synthimg, synthdepth

if __name__ == "__main__":
    cam = LytroIllum("/home/rohan/Downloads/LytroIllum_Dataset_INRIA_SIROCCO")
    cam.calibrate()
    cam.load_sensor_image(135)
    cam.decode_sensor_image(135)
    image = cam.get_decoded_image(135)
    lf = LightField(image)

    lf.show()

    disp, conf = lf.get_disparity(method='structure_tensor', fusion_method='tv_l1', epi_method = '2.5d')

    I = lf[6][6]
    D = (disp.copy()+3)/6.0
    H, W, C = lf[6][6].shape

    # create the digitally zoomed versions I1 and I2
    I1, D1 = generate_digital_zoom(I, D, zoom_factor=0.95)
    I2, D2 = I, D

    # create two separate view synthesis pipelines
    kwargs1 = {'D0': 1, 't': 0, 'u0': [H//2, W//2]}
    sp1 = SynthesisPipeline(**kwargs1)
    I1DZ, D1DZ = sp1.generate_synthesized_views(I1, D1)

    kwargs2 = {'D0': 1, 't': -0.1, 'u0': [H//2, W//2]}
    sp2 = SynthesisPipeline(**kwargs2)
    I2DZ, D2DZ = sp2.generate_synthesized_views(I2, D2)

    # image/depth fusion step
    mask = np.zeros_like(I1DZ)
    mask[np.where(I1DZ == sp1.maskf)] = 1
    dmask = np.zeros_like(D1DZ)
    dmask[np.where(D1DZ == sp1.maskf)] = 1
    I_F = mask*I2DZ + (1-mask)*I1DZ
    D_F = dmask*D2DZ + (1-dmask)*D1DZ

    # depth occlusion mask
    depth_occlusion_mask = np.zeros_like(I_F)
    depth_occlusion_mask[np.where(I_F == sp1.maskf)] = 1

    # Algorithm 1: Depth map hole filling
    H, W = D_F.shape
    D_F_bar = D_F.copy()
    for x in range(H-1):
        for y in range(W-1):
            if depth_occlusion_mask[x][y]==1:
                dmax = max(D_F[x-1][y], D_F[x][y-1], D_F[x][y+1], D_F[x+1][y])
                D_F_bar[x][y] = dmax
    
