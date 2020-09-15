#! /usr/bin/env python

from skimage import io
import argparse
import numpy as np

__doc__ = "Binarise a tif stack based on a decision threshold" \
          "used to construct the ground truth."

from skimage.morphology import flood_fill

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_file", help="TIF stack")
    parser.add_argument("threshold", type=float,
                        help="Value in [0, 1] to binarise image")

    args = parser.parse_args()

    data = io.imread(args.tif_file)
    data_out = np.array(data > args.threshold * 255, dtype=np.uint8)
    data_out = flood_fill(data_out, (1, 1, 1), 2)
    data_out_new = np.array(data_out != 2, dtype=np.uint8)

    # binarize MNS_M059_105_clahe_median_denoised_occupancy_map.tif 0.75
    # data_out_new[200:400, 512-200:512-110, 180:300] = 2

    # binarize MNS_M090_115a_clahe_median_denoised_occupancy_map.tif 0.85
    # data_out_new[225:275, 512-350:512-300, 300:350] = 2
    # data_out_new[150:200, 512-320:512-200, 280:350] = 2

    # binarize MNS_M173a_115_clahe_median_denoised_occupancy_map.tif 0.85
    # data_out_new[300:440, 512-275:512-220, 125:375] = 2
    # data_out_new[190:210, 512-170:512-140, 225:300] = 2

    # binarize MNS_M395a_115_clahe_median_denoised_occupancy_map.tif 0.85
    # data_out_new[120:210, 512-375:512-220, 300:375] = 2

    # binarize MNS_M745c_115_clahe_median_denoised_occupancy_map.tif 0.60
    # data_out_new[145:255, 512-220:512-145, 205:285] = 2

    # binarize MNS_M813_115_clahe_median_denoised_occupancy_map.tif 0.85
    # data_out_new[125:300, 512-330:512-180, 180:340] = 2

    # binarize MNS_M822_115_clahe_median_denoised_occupancy_map.tif 0.65
    # data_out_new[400:475, 512-300:512-200, 175:275] = 2
    # data_out_new[200:225, 512-240:512-160, 175:220] = 2
    # data_out_new[125:220, 512-250:512-200, 275:325] = 2

    # binarize MNS_M822a_115_clahe_median_denoised_occupancy_map.tif 0.9
    # data_out_new[370:480, 512-275:512-185, 150:275] = 2
    # data_out_new[80:175, 512-300:512-200, 200:350] = 2

    io.imsave(args.tif_file.replace(".tif", "_bin.tif"), data_out_new)

