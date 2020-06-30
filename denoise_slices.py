#! /usr/bin/env python

import argparse
import datetime
import logging
import os

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage import io, restoration
from skimage.restoration import estimate_sigma

from settings import OUT_FOLDER


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline ")

    # Argument
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run",
                        default=os.path.join(OUT_FOLDER, "denoised"))

    return parser.parse_args()


def now_string():
    """
    Return a string of the current datetime.

    :return:
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def denoise_and_save_slice(slice, slice_index, out_folder, basename):
    logging.info(f"Processing slice {slice_index}")
    plt.figure(figsize=(20, 20))
    plt.imshow(slice)

    sigma_est = np.mean(estimate_sigma(slice))
    logging.info(f"Estimated noise standard deviation = {sigma_est}")

    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6)  # 13x13 search area

    # slow algorithm
    slice_denoised = restoration.denoise_nl_means(slice,
                                                  h=1.15 * sigma_est,
                                                  fast_mode=False,
                                                  **patch_kw)

    # slice_denoised = restoration.denoise_tv_chambolle(slice)
    filename = os.path.join(out_folder, basename + "_" + str(slice_index).zfill(4) + '.png')
    plt.imsave(filename, slice_denoised)
    plt.close()


def main():
    args = parse_args()

    tif_stack_file = args.in_tif
    opt_data = io.imread(tif_stack_file)

    os.makedirs(args.out_folder, exist_ok=True)

    basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    Parallel(n_jobs=2)(delayed(denoise_and_save_slice)(slice, 286 + i, args.out_folder, basename)
                       for i, slice in enumerate(opt_data[286:, :, :]))


if __name__ == "__main__":
    main()
