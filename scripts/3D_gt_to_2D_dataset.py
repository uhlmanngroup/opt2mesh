#! /usr/bin/env python
from glob import glob

import imgaug as ia
from imgaug import augmenters as iaa

import argparse
import os

import numpy as np
import h5py
from skimage import io

__doc__ = "Create a 2D data set from 3D binary ground truth. Used to retrain a 2D UNet."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument(
        "examples_folder",
        help="Folder of HDF5 examples containing both raw and labels as 'raw' and 'labels' datasets",
    )
    parser.add_argument(
        "out_folder",
        help="Output folder for slices",
    )
    parser.add_argument(
        "--n_augment",
        type=int,
        default=5,
        help="Number of new examples to generate for a slice.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    ia.seed(1337)

    args = parse_args()
    files = sorted(glob(os.path.join(args.examples_folder, "*.h5")))

    # Pytorch-UNet format
    img_folder = os.path.join(args.out_folder, "imgs")
    mask_folder = os.path.join(args.out_folder, "masks")

    print(f"Creating folder {img_folder}")
    os.makedirs(img_folder, exist_ok=True)
    print(f"Creating folder {mask_folder}")
    os.makedirs(mask_folder, exist_ok=True)

    for f in files:
        hf = h5py.File(f, "r")
        raw = np.array(hf["raw"])
        ground_truth = np.array(hf["labels"])

        # We crop volumes eventually
        first, last = 0, 511
        raw = raw[first:last, first:last, first:last]
        ground_truth = ground_truth[first:last, first:last, first:last]

        unlabelled_class = 2

        x_indices = [i for i in range(ground_truth.shape[0]) if not(unlabelled_class in ground_truth[i, :, :])]
        y_indices = [i for i in range(ground_truth.shape[1]) if not(unlabelled_class in ground_truth[:, i, :])]
        z_indices = [i for i in range(ground_truth.shape[2]) if not(unlabelled_class in ground_truth[:, :, i])]

        print(f)
        print(f"{len(x_indices) / last * 100} % slices taken on X axis")
        print(f"{len(y_indices) / last * 100} % slices taken on Y axis")
        print(f"{len(z_indices) / last * 100} % slices taken on Z axis")

        base_name = f.split(os.sep)[-1].split(".")[0]

        print("Example shape:", raw.shape)
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

        ground_truth = ground_truth[..., np.newaxis]

        print(raw.shape, ground_truth.shape)
        examples_slices = (
            [raw[x, :, :] for x in x_indices]
            + [raw[:, y, :] for y in y_indices]
            + [raw[:, :, z] for z in z_indices]
        )

        gts_slices = (
                [ground_truth[x, :, :, :] for x in x_indices]
                + [ground_truth[:, y, :, :] for y in y_indices]
                + [ground_truth[:, :, z, :] for z in z_indices]
        )

        n_augment = args.n_augment
        files = []
        gts = []
        for i in range(n_augment):
            print(f"Augmenting step: {i} / {n_augment}")
            examples_i, gts_i = data_augmenter(
                images=examples_slices, segmentation_maps=gts_slices
            )
            files.extend(examples_i)
            gts.extend(gts_i)

        for i, (e, gt) in enumerate(zip(files, gts)):
            ex_slice_name = os.path.join(img_folder, base_name + f"_{i}.tif")
            gt_slice_name = os.path.join(mask_folder, base_name + f"_{i}.tif")
            io.imsave(ex_slice_name, e)
            io.imsave(gt_slice_name, gt)
