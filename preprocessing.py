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

import h5py
import numpy as np
from joblib import Parallel, delayed
from skimage import io, restoration, filters, exposure
from skimage.restoration import estimate_sigma
from matplotlib import pyplot as plt


def denoise_nl_means(slice):
    """
    Simple adaptation for parallelization.

    Taken and adapted from the documentation.

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


def _log_nd_array_info(data: np.ndarray):
    logging.info(f"Shape: {data.shape}")
    logging.info(f"dtype: {data.dtype}")
    logging.info(f"Min:   {data.min()}")
    logging.info(f"Max:   {data.max()}")
    logging.info(f"Median:{np.median(data)}")
    logging.info(f"Mean  :{data.mean()}")


def __parallel_denoising(joblib_parallel, opt_data, denoise_function, method):
    """
    Denoise a TIF stack a slice at a time, in a parallel fashion.

    This is not optimal: the denoising should be done on the entire stack
    but we need to have kind a lot of memory for this.

    @param joblib_parallel:
    @param opt_data:
    @param denoise_function:
    @param method:
    @return:
    """
    logging.info(f"Starting {method}")
    denoised_opt_data = joblib_parallel(delayed(denoise_function)(img)
                                        for img in opt_data)
    logging.info(f"Done with {method}")
    denoised_opt_data = np.array(denoised_opt_data)
    assert denoised_opt_data.shape == opt_data.shape

    return denoised_opt_data


def downsample(opt_data, file_basename, joblib_parallel=None):
    opt_data_downsampled = opt_data[::2, ::2, ::2]
    filename = file_basename + "_downsampled.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_downsampled.shape})")
    io.imsave(filename, opt_data_downsampled)


def denoise(opt_data, file_basename, joblib_parallel=None):
    if joblib_parallel is not None:
        with joblib_parallel:
            filename = file_basename + "_tv_denoised.tif"
            denoised_opt_data = __parallel_denoising(joblib_parallel, opt_data,
                                                     method="TV-L1 denoising (Chambolle pock)",
                                                     denoise_function=restoration.denoise_tv_chambolle)
            # Range and type conversion
            denoised_opt_data = (denoised_opt_data * 255).astype(np.uint8)
            logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
            io.imsave(filename, denoised_opt_data)

            filename = file_basename + "_median_denoised.tif"
            denoised_opt_data = __parallel_denoising(joblib_parallel, opt_data,
                                                     method="median filtering",
                                                     denoise_function=filters.median)
            logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
            io.imsave(filename, denoised_opt_data)

            filename = file_basename + "_nl_means_denoised.tif"
            denoised_opt_data = __parallel_denoising(joblib_parallel, opt_data,
                                                     method="non linear means denoising",
                                                     denoise_function=denoise_nl_means)
            logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
            io.imsave(filename, denoised_opt_data)
    else:
        filename = file_basename + "_tv_denoised.tif"
        denoised_opt_data = restoration.denoise_tv_chambolle(opt_data,
                                                             multichannel=False)
        # Range and type conversion
        denoised_opt_data = (denoised_opt_data * 255).astype(np.uint8)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = file_basename + "_median_denoised.tif"
        denoised_opt_data = filters.median(opt_data)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)

        filename = file_basename + "_nl_means_denoised.tif"
        denoised_opt_data = denoise_nl_means(opt_data)
        logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
        io.imsave(filename, denoised_opt_data)


def contrast(opt_data: np.ndarray, file_basename: str, joblib_parallel=None):
    """
    Perform contrast rectification and save results.

    The contrast equalization methods that are used are:
        - Intensity Rescaling
        - Histogram Equalization
        - Contrast Limited Adaptive Histogram Equalization

    Taken and adapted from the documentation.
    """
    logging.info("Performing contrast equalization")

    logging.info("Original data")
    _log_nd_array_info(opt_data)

    logging.info("Intensity Rescaling")
    p2, p98 = np.percentile(opt_data, (2, 98))
    opt_data_rescale = exposure.rescale_intensity(opt_data, in_range=(p2, p98))
    _log_nd_array_info(opt_data_rescale)
    filename = file_basename + "_rescaled_int.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_rescale.shape})")
    io.imsave(filename, opt_data_rescale)

    logging.info("Histogram Equalization")
    opt_data_eq = exposure.equalize_hist(opt_data)
    _log_nd_array_info(opt_data_eq)
    # Range and type conversion
    opt_data_eq = (opt_data_eq * 255).astype(np.uint8)
    _log_nd_array_info(opt_data_eq)
    filename = file_basename + "_hist_eq.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_eq.shape})")
    io.imsave(filename, opt_data_eq)

    logging.info("Contrast Limited Adaptive Histogram Equalization")
    opt_data_adapt_eq = exposure.equalize_adapthist(opt_data, clip_limit=0.03)
    # Range and type conversion
    opt_data_adapt_eq = (opt_data_adapt_eq * 255).astype(np.uint8)
    _log_nd_array_info(opt_data_adapt_eq)
    filename = file_basename + "_clahe.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_adapt_eq.shape})")
    io.imsave(filename, opt_data_adapt_eq)


def extract_png(opt_data, file_basename, joblib_parallel=None):
    """
    Extract all the slices and save them as png images.

    Typically used for visual inspection.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a png image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".png"
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(slice)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close('all')


def extract_tif(opt_data, file_basename, joblib_parallel=None):
    """
    Extract all the slices and save them as tif images.

    Typically used when using Ilastik.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".tif"
        io.imsave(filename, slice)


def crop_cube(opt_data, file_basename, joblib_parallel=None):
    """
    Crop volume. Limit determined empirically.
    """
    x_min, x_max = 0, 512
    y_min, y_max = 100, 450
    z_min, z_max = 40, 480
    croped_opt = opt_data[x_min:x_max, y_min:y_max, z_min:z_max]

    filename = file_basename + f"_{x_min}_{x_max}_{y_min}_{y_max}_{z_min}_{z_max}.tif"

    io.imsave(filename, croped_opt)


def to_hdf5(opt_data, file_basename, joblib_parallel=None):
    """
    Convert the example to hdf5
    """
    hf = h5py.File(f"{file_basename}.h5", 'w')
    hf.create_dataset("dataset", data=opt_data)
    hf.close()


commands = {func.__name__: func for func in [
    denoise,
    contrast,
    downsample,
    extract_png,
    extract_tif,
    crop_cube,
    to_hdf5
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
                        type=int, default=0)

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

    joblib_parallel = Parallel(n_jobs=args.n_jobs, backend="loky") if args.n_jobs > 0 else None

    file_basename = os.path.join(args.out_folder, basename)
    processing = commands[args.step]

    processing(opt_data, file_basename, joblib_parallel)


if __name__ == "__main__":
    main()
