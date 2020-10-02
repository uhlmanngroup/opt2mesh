#! /usr/bin/env python
import argparse

import h5py
import numpy as np

__doc__ = """
Perform the evaluation of predictions on slices which have been labelled by
computing Dice Coefficient and Intersection over Union
"""

from sklearn.metrics import adjusted_rand_score


def get_slice_indices(volume: np.ndarray, threshold: float):
    """
    Return indices of slices containing a proportion of `threshold` labels

    @param volume: the volume to inspect:
    @param threshold: proportion of slice labelled, in [0, 1]
    @return:
    """

    def _is_manually_labelled(slice: np.ndarray):
        h, w = slice.shape
        # Non labelled pixel have values 0
        proportions_labelled = np.count_nonzero(slice) / (h * w)

        return proportions_labelled >= threshold

    x_indices = [
        x for x in range(volume.shape[0]) if _is_manually_labelled(volume[x, :, :])
    ]
    y_indices = [
        y for y in range(volume.shape[1]) if _is_manually_labelled(volume[:, y, :])
    ]
    z_indices = [
        z for z in range(volume.shape[2]) if _is_manually_labelled(volume[:, :, z])
    ]

    return x_indices, y_indices, z_indices


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
        help="HDF5 file of predictions on one example (512 × 512 × 512 × n_classes) typically",
    )
    parser.add_argument(
        "ground_truth",
        help="HDF5 file of labels of one (512 × 512 × 512 × 1), last channel in {0, …, n_classes - 1}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Proportions of labels to consider a slice to be manually labelled",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    probabilities = np.array(h5py.File(args.probabilities, "r")["exported_data"])
    ground_truth = np.array(h5py.File(args.ground_truth, "r")["exported_data"])[..., 0]

    x_indices, y_indices, z_indices = get_slice_indices(
        ground_truth, threshold=args.threshold
    )

    print(f"{len(x_indices)} slices labelled on X axis: {x_indices}")
    print(f"{len(y_indices)} slices labelled on Y axis: {y_indices}")
    print(f"{len(z_indices)} slices labelled on Z axis: {z_indices}")

    # 0 indexing to labels
    predictions = np.argmax(probabilities, axis=-1) + 1
    print(predictions.shape)
    print(ground_truth.shape)

    x_dice = [dice(ground_truth[x, :, :], predictions[x, :, :]) for x in x_indices]
    y_dice = [dice(ground_truth[:, y, :], predictions[:, y, :]) for y in y_indices]
    z_dice = [dice(ground_truth[:, :, z], predictions[:, :, z]) for z in z_indices]

    x_iou = [iou(ground_truth[x, :, :], predictions[x, :, :]) for x in x_indices]
    y_iou = [iou(ground_truth[:, y, :], predictions[:, y, :]) for y in y_indices]
    z_iou = [iou(ground_truth[:, :, z], predictions[:, :, z]) for z in z_indices]

    x_rand = [rand(ground_truth[x, :, :], predictions[x, :, :]) for x in x_indices]
    y_rand = [rand(ground_truth[:, y, :], predictions[:, y, :]) for y in y_indices]
    z_rand = [rand(ground_truth[:, :, z], predictions[:, :, z]) for z in z_indices]

    print("Mean dice on X:", np.mean(x_dice))
    print("Mean dice on Y:", np.mean(y_dice))
    print("Mean dice on Z:", np.mean(z_dice))

    print("Mean IoU on X:", np.mean(x_iou))
    print("Mean IoU on Y:", np.mean(y_iou))
    print("Mean IoU on Z:", np.mean(z_iou))

    print("Mean RAND on X:", np.mean(x_rand))
    print("Mean RAND on Y:", np.mean(y_rand))
    print("Mean RAND on Z:", np.mean(z_rand))
