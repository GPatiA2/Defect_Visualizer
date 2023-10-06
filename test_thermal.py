
import cv2
from thermal import Thermal
import numpy as np
import os


dataset_dir = 'dataset_generation/datadron_real'
name = 'DJI_20220610131737_0002_T.JPG'

im    = cv2.imread(os.path.join(dataset_dir, name))
thermal = Thermal(
    dirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libdirp.so',
    dirp_sub_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_dirp.so',
    iirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_iirp.so',
    exif_filename='plugins/exiftool-12.35.exe',
    dtype=np.float32,
)
temperature = thermal.parse_dirp2(os.path.join(dataset_dir, name))
print(temperature)