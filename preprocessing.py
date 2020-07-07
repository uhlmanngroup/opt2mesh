#! /usr/bin/env python

__doc__ = """Short pipeline for the pre-processing of TIF stacks pipeline.\n\n

Can be used to perform:\n
 - contrast equalization\n
 - downsampling stack\n
 - denoising of stacks\n
 - extract all the slices\n
"""

import argparse
import logging
import os

import numpy as np
from joblib import Parallel, delayed
from skimage import io, restoration, filters, exposure
from skimage.restoration import estimate_sigma
from matplotlib import pyplot as plt


def denoise_nl_means(slice):
    """
    Simple adaptation for parallelization.
    """
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


def __parallel_denoising(joblib_parallel, opt_data, denoise_function, method):
    logging.info(f"Starting {method}")
    denoised_opt_data = joblib_parallel(delayed(denoise_function)(img)
                                        for img in opt_data)
    logging.info(f"Done with {method}")
    denoised_opt_data = np.array(denoised_opt_data)
    assert denoised_opt_data.shape == opt_data.shape

    return denoised_opt_data


def downsample(opt_data, file_basename):
    opt_data_downsampled = opt_data[::2, ::2, ::2]
    filename = file_basename + "_downsampled.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_downsampled.shape})")
    io.imsave(filename, opt_data_downsampled)


def denoise(opt_data, file_basename):
    with Parallel(n_jobs=4, backend="loky") as parallel:
        filename = file_basename + "_tv_denoised.tif"
        denoised_opt_data = __parallel_denoising(parallel, opt_data,
                                                 method="TV-L1 denoising (Chambolle pock)",
                                                 denoise_function=restoration.denoise_tv_chambolle)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = file_basename + "_median_denoised.tif"
        denoised_opt_data = __parallel_denoising(parallel, opt_data,
                                                 method="median filtering",
                                                 denoise_function=filters.median)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = file_basename + "_nl_means_denoised.tif"
        denoised_opt_data = __parallel_denoising(parallel, opt_data,
                                                 method="non linear means denoising",
                                                 denoise_function=denoise_nl_means)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)


def contrast(opt_data, file_basename):
    def log_info(data: np.ndarray):
        logging.info(f"Shape: {data.shape}")
        logging.info(f"dtype: {data.dtype}")
        logging.info(f"Min:   {data.min()}")
        logging.info(f"Max:   {data.max()}")
        logging.info(f"Median:{np.median(data)}")
        logging.info(f"Mean  :{data.mean()}")

    logging.info("Performing contrast equalization")

    logging.info("Original data")
    log_info(opt_data)

    logging.info("Contrast stretching")
    p2, p98 = np.percentile(opt_data, (2, 98))
    opt_data_rescale = exposure.rescale_intensity(opt_data, in_range=(p2, p98))
    log_info(opt_data_rescale)
    filename = file_basename + "_rescale.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_rescale.shape})")
    io.imsave(filename, opt_data_rescale)

    logging.info("Equalization")
    opt_data_eq = exposure.equalize_hist(opt_data)
    log_info(opt_data_eq)
    opt_data_eq = (opt_data_eq * 255).astype(np.uint8)
    log_info(opt_data_eq)
    filename = file_basename + "_eq.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_eq.shape})")
    io.imsave(filename, opt_data_eq)

    logging.info("Adaptive Equalization")
    opt_data_adapt_eq = exposure.equalize_adapthist(opt_data, clip_limit=0.03)
    log_info(opt_data_adapt_eq)
    filename = file_basename + "_adapt_eq.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_adapt_eq.shape})")
    io.imsave(filename, opt_data_adapt_eq)


def extract_png(opt_data, file_basename):
    """
    Extract all the slices and save them as png images.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a png image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".png"
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(slice)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close('all')


def extract_tif(opt_data, file_basename):
    """
    Extract all the slices and save them as tif images.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".tif"
        io.imsave(filename, slice)


commands = {func.__name__: func for func in [
    denoise,
    contrast,
    downsample,
    extract_png,
    extract_tif
]}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__)

    # Argument
    parser.add_argument("step", help="Pre-processing step to perform",
                        choices=list(commands.keys()), default="contrast")
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run")

    parser.add_argument("--n_jobs", help="Number of parallel jobs",
                        type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    tif_stack_file = args.in_tif
    opt_data = io.imread(tif_stack_file)

    os.makedirs(args.out_folder, exist_ok=True)

    # Convert a path like '/path/to/file.name.ext' to 'file.name'
    basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    file_basename = os.path.join(args.out_folder, basename)
    processing = commands[args.step]

    processing(opt_data, file_basename)


if __name__ == "__main__":
    main()
