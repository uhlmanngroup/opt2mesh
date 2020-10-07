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

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import io, restoration, filters, exposure
from skimage.morphology import erosion, dilation
from skimage.restoration import estimate_sigma


def _empirical_crop(volume):
    """
    Limit were determined empirically.

    @param volume:
    @return:
    """
    x_min, x_max = 0, 512
    y_min, y_max = 100, 450
    z_min, z_max = 40, 480

    coords = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

    return volume[x_min:x_max, y_min:y_max, z_min:z_max], coords


def __denoise_nl_means(slice):
    """
    Simple adaptation for parallelization.

    Taken and adapted from the documentation.
    """
    sigma_est = np.mean(estimate_sigma(slice))

    patch_kw = dict(patch_size=5, patch_distance=6)  # 5x5 patches  # 13x13 search area

    # slow algorithm
    slice_denoised = restoration.denoise_nl_means(
        slice, h=1.15 * sigma_est, fast_mode=False, preserve_range=True, **patch_kw
    )

    return slice_denoised


def _log_nd_array_info(data: np.ndarray):
    logging.info(f"Shape: {data.shape}")
    logging.info(f"dtype: {data.dtype}")
    logging.info(f"Min:   {data.min()}")
    logging.info(f"Max:   {data.max()}")
    logging.info(f"Median:{np.median(data)}")
    logging.info(f"Mean  :{data.mean()}")


def downsample(opt_data, file_basename):
    """
    Downsample the data by a factor of 2.

    @param opt_data:
    @param file_basename:
    @return:
    """
    opt_data_downsampled = opt_data[::2, ::2, ::2]
    filename = file_basename + "_downsampled.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_downsampled.shape})")
    io.imsave(filename, opt_data_downsampled)


def denoise(opt_data, file_basename):
    """
    Perform denoising and save results.

    The denoising methods that are used are:
        - TV-L1 denoising (Chambolle pock)
        - Median filtering
        - non linear means denoising

    Taken and adapted from skimage's documentation.
    @param opt_data:
    @param file_basename:
    @return:
    """
    filename = file_basename + "_median_denoised.tif"
    denoised_opt_data = filters.median(opt_data)
    logging.info(f"Saving at {filename} (shape: {denoised_opt_data.shape})")
    io.imsave(filename, denoised_opt_data)


def contrast(opt_data: np.ndarray, file_basename: str):
    """
    Perform contrast rectification and save results.

    The contrast equalization methods that are used are:
        - Intensity Rescaling
        - Histogram Equalization
        - Contrast Limited Adaptive Histogram Equalization

    Taken and adapted from skimage's documentation.
    """
    logging.info("Performing contrast equalization")

    logging.info("Original data")
    _log_nd_array_info(opt_data)

    logging.info("Contrast Limited Adaptive Histogram Equalization")
    opt_data_adapt_eq = exposure.equalize_adapthist(opt_data, clip_limit=0.03)
    # Range and type conversion
    opt_data_adapt_eq = (opt_data_adapt_eq * 255).astype(np.uint8)
    _log_nd_array_info(opt_data_adapt_eq)
    filename = file_basename + "_clahe.tif"
    logging.info(f"Saving at {filename} (shape: {opt_data_adapt_eq.shape})")
    io.imsave(filename, opt_data_adapt_eq)


def extract_png(opt_data, file_basename):
    """
    Extract all the slices and save them as png images.

    Typically used for visual inspection.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a png image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".png"
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(slice)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close("all")


def extract_tif(opt_data, file_basename):
    """
    Extract all the slices and save them as tif images.
    """
    for slice_index, slice in enumerate(opt_data):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_" + str(slice_index).zfill(4) + ".tif"
        io.imsave(filename, slice)


def extract_tif_x(opt_data, file_basename):
    """
    Extract all the slices on the x axis and save them as tif images.
    """
    h, w, d = opt_data.shape

    for slice_index in range(h):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_x_" + str(slice_index).zfill(4) + ".tif"
        slice = opt_data[slice_index, :, :]
        io.imsave(filename, slice)


def extract_tif_y(opt_data, file_basename):
    """
    Extract all the slices on the y axis and save them as tif images.
    """
    h, w, d = opt_data.shape

    for slice_index in range(w):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_y_" + str(slice_index).zfill(4) + ".tif"
        slice = opt_data[:, slice_index, :]
        io.imsave(filename, slice)


def extract_tif_z(opt_data, file_basename):
    """
    Extract all the slices on the z axis and save them as tif images.
    """
    h, w, d = opt_data.shape

    for slice_index in range(d):
        logging.info(f"Extracting slice {slice_index} as a tif image")
        filename = file_basename + "_z_" + str(slice_index).zfill(4) + ".tif"
        slice = opt_data[:, :, slice_index]
        io.imsave(filename, slice)


def crop_cube(opt_data, file_basename):
    """
    Crop volume. Limit determined empirically.
    """
    cropped_opt, coords = _empirical_crop(opt_data)
    x_min, x_max = coords[0]
    y_min, y_max = coords[1]
    z_min, z_max = coords[2]
    croped_opt = opt_data[x_min:x_max, y_min:y_max, z_min:z_max]

    filename = file_basename + f"_{x_min}_{x_max}_{y_min}_{y_max}_{z_min}_{z_max}.tif"

    io.imsave(filename, croped_opt)


def to_hdf5(opt_data, file_basename):
    """
    Convert the example to hdf5
    """
    file_name = f"{file_basename}.h5"
    hf = h5py.File(file_name, "w")
    hf.create_dataset("dataset", data=opt_data, chunks=True)
    hf.close()

    return file_name


def full(opt_data, file_basename):
    """
    Perform the full preprocessing of the data.

    @param opt_data:
    @param file_basename: for the export
    @return:
    """

    logging.info(f"Performing full preprocessing on {file_basename.split(os.sep)[-1]}")
    _log_nd_array_info(opt_data)

    logging.info("Contrast Limited Adaptive Histogram Equalization")
    opt_data_adapt_eq = exposure.equalize_adapthist(opt_data, clip_limit=0.03)
    opt_data_adapt_eq = (opt_data_adapt_eq * 255).astype(np.uint8)
    _log_nd_array_info(opt_data_adapt_eq)
    filename = file_basename + "_clahe"
    _log_nd_array_info(opt_data)

    logging.info("Median filtering")
    denoised_opt_data = filters.median(opt_data_adapt_eq)
    filename += "_median_denoised"

    logging.info(f"Cropping volume")
    x_min, x_max = 0, 512
    y_min, y_max = 100, 450
    z_min, z_max = 40, 480
    logging.info(f"  x: {x_min}, {x_max}")
    logging.info(f"  y: {y_min}, {y_max}")
    logging.info(f"  z: {z_min}, {z_max}")
    cropped_opt = denoised_opt_data[x_min:x_max, y_min:y_max, z_min:z_max]
    filename += f"_{x_min}_{x_max}_{y_min}_{y_max}_{z_min}_{z_max}.tif"

    logging.info(f"Saving as tif: {filename}")
    io.imsave(filename, cropped_opt)


def _fill_binary_image(im_slice):
    # Copy the thresholded image.
    im_floodfill = im_slice.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_slice.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_slice | im_floodfill_inv

    return im_out


def _morphological_post_processing(im_slice):
    """
    Experimental: clean the segmentation using morphological operations.

    @param im_slice: 2D slice images
    @return:
    """
    erode_shape = (3, 3)
    dilate_shape = (3, 3)
    postprocessed_slice = (
        cv2.dilate(
            cv2.erode(
                cv2.GaussianBlur(im_slice, ksize=(3, 3), sigmaX=1, sigmaY=1),
                np.ones(erode_shape),
            ),
            np.ones(dilate_shape),
        )
        > 255 / 2
    ).astype(np.uint8)

    return postprocessed_slice


def _post_process_binary_slice(im_slice, n_step=1):
    """
    Perform some erosion and dilation and then.

    Fill the inside of a binary image.

    @param im_slice: 2D slice images
    @return:
    """
    im_out = im_slice
    for _ in range(n_step):
        im_out = _morphological_post_processing(_fill_binary_image(im_out))

    return im_out


def clean_seg(segmentation_data, file_basename):
    """
    Experimental: clean the segmentation using morphological operations.

    @param segmentation_data:
    @param file_basename: for the export
    @return:
    """
    improved_seg_data = dilation(
        erosion(dilation(gaussian_filter(segmentation_data, sigma=0.1)))
    ).astype(np.uint8)
    for i in range(segmentation_data.shape[0]):
        improved_seg_data[i, :, :] = _fill_binary_image(improved_seg_data[i, :, :])
    filename = file_basename + f"_cleaned.tif"

    io.imsave(filename, improved_seg_data)


commands = {
    func.__name__: func
    for func in [
        denoise,
        contrast,
        downsample,
        extract_png,
        extract_tif,
        extract_tif_x,
        extract_tif_y,
        extract_tif_z,
        crop_cube,
        to_hdf5,
        clean_seg,
        full,
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument(
        "step",
        help="Pre-processing step to perform",
        choices=list(commands.keys()),
        default="contrast",
    )
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run")

    return parser.parse_args()


def main():
    args = parse_args()

    tif_stack_file = args.in_tif

    if tif_stack_file.endswith("tif"):
        opt_data = io.imread(tif_stack_file)
    else:
        f = h5py.File(tif_stack_file, "r")
        key = list(f.keys())[0]
        opt_data = np.array(f[key])
        f.close()

    os.makedirs(args.out_folder, exist_ok=True)

    # Convert a path like '/path/to/file.name.ext' to 'file.name'
    basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    file_basename = os.path.join(args.out_folder, basename)
    processing = commands[args.step]

    processing(opt_data, file_basename)


if __name__ == "__main__":
    main()
