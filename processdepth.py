from plenpy.lightfields import LightField
from plenpy.cameras.lytro_illum import LytroIllum
from pylab import *

cam = LytroIllum("/home/rohan/Downloads/LytroIllum_Dataset_INRIA_SIROCCO")
cam.calibrate()
cam.load_sensor_image(136)
cam.decode_sensor_image(136)
image = cam.get_decoded_image(136)
lf = LightField(image)

lf.show()

disp, conf = lf.get_disparity(method='structure_tensor', fusion_method='tv_l1', epi_method = '2.5d')
