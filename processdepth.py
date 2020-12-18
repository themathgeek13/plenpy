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
import cv2

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

MASK = 10

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)

    Reference: https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/37121993#37121993
    """

    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def depth_map_hole_filling(D_F, depth_occlusion_mask):
    H, W = D_F.shape
    D_F_bar = D_F.copy()
    for x in range(H-1):
        for y in range(W-1):
            if depth_occlusion_mask[x][y]==1:
                dmax = max(D_F[x-1][y], D_F[x][y-1], D_F[x][y+1], D_F[x+1][y])
                D_F_bar[x][y] = dmax
    return D_F_bar

def discretedepth(D, N=20):
    discdepth = D.copy()
    u = np.unique(D).reshape(-1,1)
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(u)
    intervals = [1]+sorted(kmeans.cluster_centers_, reverse=True)+[0]
    for i in range(len(intervals)-1):
        pos = np.where((discdepth<=intervals[i]) & (discdepth>intervals[i+1]))
        discdepth[pos] = intervals[i]

    return discdepth 

def image_hole_filling(I_F, D_F_bar_disc, M):
    # step 1
    finalimg = []
    for ch in range(3):
        I_F_bar = np.copy(I_F[:,:,ch])
        # step 2
        d_u = np.unique(D_F_bar_disc)
        S = len(d_u)
        M_prev = np.zeros_like(M[:,:,ch])

        # step 3
        for s in range(S-1, 0, -1):
            # 3.1
            pos = np.where((D_F_bar_disc > d_u[s-1]) & (D_F_bar_disc <= d_u[s]))
            D_s = np.zeros_like(D_F_bar_disc, dtype=np.float)
            D_s[pos] = 1

            # 3.2
            I_s = np.multiply(I_F[:,:,ch], D_s)

            # 3.3
            M_curr = np.multiply(M[:,:,ch], D_s)

            # 3.4
            M_curr = np.logical_or(M_curr, M_prev)

            # 3.5
            for x in range(H-1):
                for y in range(W-1):
                    if M_curr[x,y]==1:
                        # 3.5.1
                        # find nearest valid pixel in same row
                        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
                        def find_nearest_non_zero(array, value):
                            posarr = array[np.where(array>0)]
                            if len(posarr) == 0:
                                return 0, 0
                            idx = (np.abs(posarr-value)).argmin()
                            val = posarr[idx]
                            # print(np.where(abs(array-val)<1e-6)[0][0])
                            mainidx = int(np.where(abs(array-val)<1e-6)[0][0])
                            return mainidx, val
                        row = M[:,y]
                        if x > 0:
                            revidx, __ = find_nearest_non_zero(row[:x][::-1], 0)
                            idx, __ = find_nearest_non_zero(row[x:], 0)
                            if(revidx < idx):
                                xd = x - revidx - 1
                            else:
                                xd = x + idx
                        else:
                            xd, __ = find_nearest_non_zero(row, 0)

                        # 3.5.2 update the value of I_F_bar
                        # print(x,xd)
                        I_F_bar[x][y] = I_s[xd][y]
                        
                        # 3.5.3 update M_curr
                        M_curr[x][y] = 0

                        # 3.5.4 update M
                        M[x][y][ch] = 0

            # 3.6 propagate current occlusion mask
            M_prev = M_curr.copy()
        
        # apply simple low pass filtering on the filled-in 
        # occluded areas in I_F_bar
        finalimg.append(cv2.medianBlur(I_F_bar, 5))
    
    I_F_bar = np.dstack(finalimg)
    return I_F_bar

def shallow_depth_of_field(D_F_bar_disc, I_F_bar, DISCRETE_DEPTH, PREF_DEPTH):
    # get a set of blur values for depths
    sigma_map = np.array(np.round(abs(np.array(D_F_bar_disc, dtype='float32') - PREF_DEPTH)*255).clip(0,255), dtype='float')
    # convert them to 0 to 7
    blur_kernels = np.array(np.unique(sigma_map)*11/np.max(sigma_map), dtype='uint8')
    blur_kernels = np.ceil(blur_kernels) // 2 * 2 + 1
    print(blur_kernels)

    d_u = np.unique(D_F_bar_disc)
    I_F_bar_DZ = np.zeros_like(I_F_bar)
    for depth in range(DISCRETE_DEPTH+2):
        # get image segments at that depth:
        depthsegmask = np.ones_like(I_F_bar)
        condition = np.where(abs(D_F_bar_disc - d_u[depth])<1e-3)
        depthsegmask[condition] = 2
        depthsegmask/=2
        # dilate the mask a bit
        # kernel = np.ones((3,3),np.uint8)
        # depthsegmask = cv2.morphologyEx(depthsegmask, cv2.MORPH_CLOSE, kernel)
        depthsegimg = depthsegmask * I_F_bar
        # plt.imshow(depthsegmask)
        # plt.pause(1)

        # now blur this map by using a kernel of that size
        bksize = blur_kernels[depth]
        blurimg = cv2.GaussianBlur(depthsegimg, (bksize, bksize), 0)

        # now use this blurred image and replace original image with it
        I_F_bar_DZ += blurimg*depthsegmask

        # plt.imshow(I_F_bar_DZ)
        # plt.pause(1)

    I_F_bar_DZ *= I_F_bar.max()/I_F_bar_DZ.max()

    finaloutput = cv2.medianBlur(I_F_bar_DZ, 5)
    return finaloutput

def generate_synthesized_views(inputimg, inputdepth, t, D0):
    x1A, y1A, __ = np.indices(inputimg.shape)
    H, W, C = inputimg.shape
    u0 = [H//2, W//2]
    # forward warping(f, h) -> g:
    #   for every pixel x in f(x):
    #       1. compute destination location x' = h(x)
    #       2. copy pixel from f(x) to g(x')
    resultimg = np.ones_like(inputimg)*MASK
    zbuffer = np.ones_like(inputdepth)*MASK
    for x in range(W):
        for y in range(H):
            D1A = inputdepth[y][x]
            newx = np.clip(int(D1A*(D0-t)*x/D0/(D1A-t+1e-3) + t*(D1A-D0)*u0[1]/D0/(D1A-t+1e-3)), 0, W-1)
            newy = np.clip(int(D1A*(D0-t)*y/D0/(D1A-t+1e-3) + t*(D1A-D0)*u0[0]/D0/(D1A-t+1e-3)), 0, H-1)
            if inputdepth[newy][newx] < zbuffer[newy][newx]:
                zbuffer[newy][newx] = inputdepth[newy][newx]
                resultimg[newy][newx] = inputimg[y][x]
    return resultimg, zbuffer

def run_single_shot_pipeline(I, D, t):
    # create the digitally zoomed versions I1 and D1
    I1 = cv2_clipped_zoom(I, 1.0/(1.0-abs(t)))
    D1 = cv2_clipped_zoom(D, 1.0/(1.0-abs(t)))
    I2, D2 = I, D

    # create two separate view synthesis pipelines
    I1DZ, D1DZ = generate_synthesized_views(I1, D1, t/2, 1.0) #sp1.generate_synthesized_views(I1, D1)

    I2DZ, D2DZ = generate_synthesized_views(I2, D2, t/2, 1.0) #sp2.generate_synthesized_views(I2, D2)

    # image/depth fusion step
    mask = np.zeros_like(I1DZ)
    mask[np.where(I1DZ == MASK)] = 1
    dmask = np.zeros_like(D1DZ)
    dmask[np.where(D1DZ == MASK)] = 1
    I_F = mask*I2DZ + (1-mask)*I1DZ
    D_F = dmask*D2DZ + (1-dmask)*D1DZ

    I_F = np.asarray(I_F)
    D_F = np.asarray(D_F)

    # depth occlusion mask
    depth_occlusion_mask = np.zeros_like(D_F)
    depth_occlusion_mask[np.where(D_F == MASK)] = 1

    # image occlusion mask
    M = np.zeros_like(np.asarray(I_F))
    M[np.where(I_F == MASK)] = 1

    # Algorithm 1: Depth map hole filling
    D_F_bar = depth_map_hole_filling(D_F, depth_occlusion_mask)

    # set the remaining points to near-zero, since we need 
    # to use the depth map after this
    D_F_bar[np.where(D_F_bar == MASK)] = 1e-3

    # Need to discretize the depth values for the next algorithm
    # Depth values always in the range of -1 to 1
    # But since it is often a subject based shot, many values clustered
    # around the center, meaning Gaussian fit would be best
    DISCRETE_DEPTH = 10
    PREF_DEPTH = 0.9
    D_F_bar_disc = discretedepth(D_F_bar, N=DISCRETE_DEPTH)

    print("INFO: Done discretizing the depth values.")
        
    # Algorithm 2: Image hole filling
    I_F_bar = image_hole_filling(I_F, D_F_bar_disc, M)
    I_F_bar[np.where(I_F_bar > MASK/2)] = 0

    # Algorithm 3: Shallow Depth of Field
    finalresult = shallow_depth_of_field(D_F_bar_disc, I_F_bar.copy(), DISCRETE_DEPTH, PREF_DEPTH)

    return I1, D1, I2, D2, I1DZ, D1DZ, I2DZ, D2DZ, I_F, D_F, I_F_bar, D_F_bar, finalresult

if __name__ == "__main__":
    cam = LytroIllum("/home/rohan/Downloads/LytroIllum_Dataset_INRIA_SIROCCO")
    cam.calibrate()
cam.load_sensor_image(310)
cam.decode_sensor_image(310)
image = cam.get_decoded_image(310)
lf = LightField(image)

# lf.show()

disp, conf = lf.get_disparity(method='structure_tensor', fusion_method='tv_l1', epi_method = '2.5d')

I = lf[6][6]
D = (disp.copy()+3)/6.0
H, W, C = lf[6][6].shape

for i in range(10,11):
    I1, D1, I2, D2, I1DZ, D1DZ, I2DZ, D2DZ, I_F, D_F, I_F_bar, D_F_bar, finaloutput = run_single_shot_pipeline(I, D, t=0.01*(i+1))
    plt.imsave(str(i+1)+".png", finaloutput)