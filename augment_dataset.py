#! /usr/bin/env python

import imgaug as ia
from imgaug import augmenters as iaa

import argparse
import os

import numpy as np
import h5py
from skimage import io

from segmentation_evaluation import get_slice_indices

__doc__ = "2D dataset creation from 3D images"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("example",
                        help="HDF5 file of one example (512 × 512 × 512 × n_classes) typically")
    parser.add_argument("ground_truth",
                        help="HDF5 file of labels of one (512 × 512 × 512 × 1), last channel in {0, …, n_classes}")
    parser.add_argument("out_folder",
                        help="Name of the output folder for slices")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Proportions of labels to consider a slice to be manually labelled")

    return parser.parse_args()


if __name__ == "__main__":
    ia.seed(1337)

    args = parse_args()

    example = np.array(h5py.File(args.example, "r")["dataset"])
    ground_truth = np.array(h5py.File(args.ground_truth, "r")["exported_data"])[..., 0]

    x_indices, y_indices, z_indices = get_slice_indices(ground_truth,
                                                        threshold=args.threshold)

    print(args.example)
    print(f"{len(x_indices)} slices labelled on X axis: {x_indices}")
    print(f"{len(y_indices)} slices labelled on Y axis: {y_indices}")
    print(f"{len(z_indices)} slices labelled on Z axis: {z_indices}")

    # Pytorch-UNet format
    img_folder = os.path.join(args.out_folder, "imgs")
    mask_folder = os.path.join(args.out_folder, "masks")

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    base_name = args.example.split(os.sep)[-1].split(".h5")[0]

    # We crop examples
    first = 0
    last = 511

    # Changing merging background and unlabelled slices together by default
    ground_truth[ground_truth == 2] = 0

    # Inverting
    ground_truth = (1 - ground_truth)[:, np.newaxis]

    # Data Augmenter
    data_augmenter = iaa.Sequential([
        iaa.PiecewiseAffine(scale=(0.01, 0.03)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.4))
    ], random_order=True)

    n_augment = 5

    for x in x_indices:
        example_slice = example[x, :, :][first:last, first:last]
        ground_truth_slice = ground_truth[x, :, :][first:last, first:last]
        ex_slice_name = os.path.join(img_folder, base_name + f"_x_{x}.tif")
        gt_slice_name = os.path.join(mask_folder, base_name + f"_x_{x}.tif")
        io.imsave(ex_slice_name, example_slice)
        io.imsave(gt_slice_name, ground_truth_slice)

        for i in range(n_augment):
            example_slice_mod, ground_truth_slice_mod =\
                data_augmenter(images=[example_slice], segmentation_maps=[ground_truth_slice])
            ex_slice_name = os.path.join(img_folder, base_name + f"_x_{x}_{i}.tif")
            gt_slice_name = os.path.join(mask_folder, base_name + f"_x_{x}_{i}.tif")
            io.imsave(ex_slice_name, example_slice_mod)
            io.imsave(gt_slice_name, ground_truth_slice_mod)

    for y in y_indices:
        example_slice = example[:, y, :][first:last, first:last]
        ground_truth_slice = ground_truth[:, y, :][first:last, first:last]
        ex_slice_name = os.path.join(img_folder, base_name + f"_y_{y}.tif")
        gt_slice_name = os.path.join(mask_folder, base_name + f"_y_{y}.tif")
        io.imsave(ex_slice_name, example_slice)
        io.imsave(gt_slice_name, ground_truth_slice)

        for i in range(n_augment):
            example_slice_mod, ground_truth_slice_mod =\
                data_augmenter(images=[example_slice], segmentation_maps=[ground_truth_slice])
            ex_slice_name = os.path.join(img_folder, base_name + f"_y_{y}_{i}.tif")
            gt_slice_name = os.path.join(mask_folder, base_name + f"_y_{y}_{i}.tif")
            io.imsave(ex_slice_name, example_slice_mod)
            io.imsave(gt_slice_name, ground_truth_slice_mod)

    for z in z_indices:
        example_slice = example[:, :, z][first:last, first:last]
        ground_truth_slice = ground_truth[:, :, z][first:last, first:last]
        ex_slice_name = os.path.join(img_folder, base_name + f"_z_{z}.tif")
        gt_slice_name = os.path.join(mask_folder, base_name + f"_z_{z}.tif")
        io.imsave(ex_slice_name, example_slice)
        io.imsave(gt_slice_name, ground_truth_slice)

        for i in range(n_augment):
            example_slice_mod, ground_truth_slice_mod =\
                data_augmenter(images=[example_slice], segmentation_maps=[ground_truth_slice])
            ex_slice_name = os.path.join(img_folder, base_name + f"_y_{z}_{i}.tif")
            gt_slice_name = os.path.join(mask_folder, base_name + f"_y_{z}_{i}.tif")
            io.imsave(ex_slice_name, example_slice_mod)
            io.imsave(gt_slice_name, ground_truth_slice_mod)
