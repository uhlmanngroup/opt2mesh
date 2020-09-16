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
    parser.add_argument(
        "example",
        help="HDF5 file of one example (512 × 512 × 512 × n_classes) typically",
    )
    parser.add_argument(
        "ground_truth",
        help="HDF5 file of labels of one (512 × 512 × 512 × 1), last channel in {0, …, n_classes}",
    )
    parser.add_argument("out_folder", help="Name of the output folder for slices")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="proportions of labels to consider a slice to be manually labelled",
    )
    parser.add_argument(
        "--n_augment",
        type=int,
        default=50,
        help="Number of new examples to generate from the data" "that we have.",
    )
    parser.add_argument(
        "--old_labelling",
        help="Adapt the augmentation for 'background/border/embryo' instead",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    ia.seed(1337)

    args = parse_args()

    if args.example.endswith(".h5"):
        example = np.array(h5py.File(args.example, "r")["dataset"])
    elif args.example.endswith(".tif"):
        example = io.imread(args.example)

    ground_truth = np.array(h5py.File(args.ground_truth, "r")["exported_data"])[..., 0]

    # We crop examples
    first, last = 0, 511

    example = example[first:last, first:last, first:last]
    ground_truth = ground_truth[first:last, first:last, first:last]

    x_indices, y_indices, z_indices = get_slice_indices(
        ground_truth, threshold=args.threshold
    )

    print(args.example)
    print(f"{len(x_indices)} slices labelled on X axis: {x_indices}")
    print(f"{len(y_indices)} slices labelled on Y axis: {y_indices}")
    print(f"{len(z_indices)} slices labelled on Z axis: {z_indices}")

    # Pytorch-UNet format
    img_folder = os.path.join(args.out_folder, "imgs")
    mask_folder = os.path.join(args.out_folder, "masks")

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    base_name = args.example.split(os.sep)[-1].split(".")[0]

    if args.old_labelling:
        # Mapping the labelling:
        #    0: unlabelled
        #    1: background
        #    2: border
        #    3: embryo
        # To a binary labelling:
        #    0: background
        #    1: embryo

        # Merging embryo and unlabelled slices together by default
        ground_truth[ground_truth == 3] = 2
        ground_truth[ground_truth == 2] = 0
        # Inverting 0 and 1
        ground_truth = (1 - ground_truth)[..., np.newaxis]

    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth[..., np.newaxis]
    print("Example shape:", example.shape)
    print("Ground truth shape:", ground_truth.shape)

    # Data Augmenter
    data_augmenter = iaa.Sequential(
        [
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.Fliplr(0.5),
        ],
        random_order=True,
    )

    print(example.shape, ground_truth.shape)
    examples_slices = (
        [example[x, :, :] for x in x_indices]
        + [example[:, y, :] for y in y_indices]
        + [example[:, :, z] for z in z_indices]
    )

    gts_slices = (
        [ground_truth[x, :, :, :] for x in x_indices]
        + [ground_truth[:, y, :, :] for y in y_indices]
        + [ground_truth[:, :, z, :] for z in z_indices]
    )

    n_augment = args.n_augment
    examples = []
    gts = []
    for i in range(n_augment):
        print(f"Augmenting step: {i} / {n_augment}")
        examples_i, gts_i = data_augmenter(
            images=examples_slices, segmentation_maps=gts_slices
        )
        examples.extend(examples_i)
        gts.extend(gts_i)

    for i, (e, gt) in enumerate(zip(examples, gts)):
        ex_slice_name = os.path.join(img_folder, base_name + f"_{i}.tif")
        gt_slice_name = os.path.join(mask_folder, base_name + f"_{i}.tif")
        io.imsave(ex_slice_name, e)
        io.imsave(gt_slice_name, gt)