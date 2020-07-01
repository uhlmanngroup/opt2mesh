#! /usr/bin/env python

import argparse
import datetime
import logging
import os

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage import io, restoration, filters
from skimage.restoration import estimate_sigma

from settings import OUT_FOLDER


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline ")

    # Argument
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run",
                        default=os.path.join(OUT_FOLDER, "denoised"))

    parser.add_argument("--n_jobs", help="Number of parallel jobs",
                        type=int, default=4)

    return parser.parse_args()


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

    filename = os.path.join(out_folder, basename + "_" + str(slice_index).zfill(4) + '.png')
    plt.imsave(filename, slice_denoised)
    plt.close()


def denoise_nl_means(slice):
    sigma_est = np.mean(estimate_sigma(slice))

    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6)  # 13x13 search area

    # slow algorithm
    slice_denoised = restoration.denoise_nl_means(slice,
                                                  h=1.15 * sigma_est,
                                                  fast_mode=False,
                                                  preserve_range=True,
                                                  **patch_kw)

    return slice_denoised


def parallel_denoising(joblib_parallel, opt_data, denoise_function, method):
    logging.info(f"Starting {method}")
    denoised_opt_data = joblib_parallel(delayed(denoise_function)(img)
                                        for img in opt_data)
    logging.info(f"Done with {method}")
    denoised_opt_data = np.array(denoised_opt_data)
    assert denoised_opt_data.shape == opt_data.shape

    return denoised_opt_data


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

    with Parallel(n_jobs=args.n_jobs, backend="loky") as parallel:
        filename = os.path.join(args.out_folder, basename + "_tv_denoised.tif")
        denoised_opt_data = parallel_denoising(parallel, opt_data,
                                               method="TV-L1 denoising (Chambolle pock)",
                                               denoise_function=restoration.denoise_tv_chambolle)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = os.path.join(args.out_folder, basename + "_median_denoised.tif")
        denoised_opt_data = parallel_denoising(parallel, opt_data,
                                               method="median filtering",
                                               denoise_function=filters.median)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = os.path.join(args.out_folder, basename + "_nl_means_denoised.tif")
        denoised_opt_data = parallel_denoising(parallel, opt_data,
                                               method="non linear means denoising",
                                               denoise_function=denoise_nl_means)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)


if __name__ == "__main__":
    main()
