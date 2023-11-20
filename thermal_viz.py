import cv2
from PIL import Image
import argparse
import os
from thermal import Thermal
import numpy as np
from matplotlib import pyplot as plt

def options():
    parser = argparse.ArgumentParser(description='Thermal visualization')
    parser.add_argument('--dataset_dir', type=str, default='dataset_generation/datadron_real')
    parser.add_argument('--name', type=str, default='DJI_20220610133530_0627_T.JPG')
    args = parser.parse_args()
    return args

args = options()
dataset_dir = args.dataset_dir
name = args.name

im    = cv2.imread(os.path.join(dataset_dir, name))

thermal = Thermal(
    dirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libdirp.so',
    dirp_sub_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_dirp.so',
    iirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_iirp.so',
    exif_filename='plugins/exiftool-12.35.exe',
    dtype=np.float32,
)

temperature = thermal.parse_dirp2(os.path.join(dataset_dir, name))

print(temperature.shape)

fig = plt.hist(temperature.flatten(), bins = 510 )

print(np.sum(fig[0]))

plt.show()