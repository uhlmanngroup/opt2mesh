#! /usr/bin/env python
import argparse
import os
from glob import glob

import numpy as np

__doc__ = """
Perform the evaluation of predictions on volumes which have been labelled by
computing Dice Coefficient and Intersection over Union. Doesn't use the unlabelled class (2).
"""

from skimage import io

from sklearn.metrics import adjusted_rand_score


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def iou(im1, im2):
    """
    Computes the Intersection over Union coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    iou : float
        Intersection over Union coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    return np.sum(intersection) / np.sum(union)


def rand(im1, im2):
    labels_true = np.ndarray.flatten(im1)
    labels_pred = np.ndarray.flatten(im2)
    return adjusted_rand_score(labels_true, labels_pred)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument(
        "probabilities",
        help="Folder of TIF files predictions of (511 × 511 × 511) examples",
    )
    parser.add_argument(
        "ground_truths",
        help="Folder of TIF files of labels of (511 × 511 × 511) examples, values in [0, 1, 2]",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    probs_files = sorted(glob(os.path.join(args.probabilities, "*.tif")))
    gt_files = sorted(glob(os.path.join(args.ground_truths, "*.tif")))

    print(f"Number of files for predictions : {len(probs_files)}")
    print(f"Number of files for ground truth: {len(gt_files)}")

    assert len(probs_files) == len(gt_files), "Different number of files in each folder"

    for pf, gt in zip(probs_files, gt_files):

        probabilities = io.imread(pf)
        ground_truth = io.imread(gt)

        if len(probabilities.shape) == 4:
            probabilities = probabilities[probabilities.shape.index(1)]

        if len(ground_truth.shape) == 4:
            ground_truth = ground_truth[ground_truth.shape.index(1)]

        assert probabilities.shape == (511, 511, 511)
        assert ground_truth.shape == (511, 511, 511)

        predictions = np.array(probabilities > 0.5, dtype=np.uint8)

        # Do not consider the "unlabelled" class voxels
        unlabelled_class = 2
        mask = (ground_truth != unlabelled_class)

        print(pf, "vs", gt)
        print("Dice          :", dice(ground_truth[mask], predictions[mask]))
        print("IoU           :", iou(ground_truth[mask], predictions[mask]))
        print()
