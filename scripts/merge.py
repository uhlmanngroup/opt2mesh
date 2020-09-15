#! /usr/bin/env python

import sys
import argparse
from skimage import io
import h5py
import os
import numpy as np

__doc__ = "Merge examples and their labels together in the 3D UNet format"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("examples_folder")
    parser.add_argument("predictions_folder")
    parser.add_argument("out_folder")
    parser.add_argument("--prediction_level", help="Decision threshold for"
                                                   "binarization", type=float, default=0.85)

    args = parser.parse_args()

    examples_files = sorted(os.listdir(args.examples_folder))
    predictions_files = sorted(os.listdir(args.predictions_folder))

    os.makedirs(args.out_folder, exist_ok=True)

    for ex_f, prediction_f in zip(examples_files, predictions_files):
        ex = io.imread(os.path.join(args.examples_folder, ex_f))
        prediction = io.imread(os.path.join(args.predictions_folder,
                                            prediction_f))

        print("Processing", ex_f, "and", prediction_f)

        assert 0 <= ex.min(), "Should be in [[0, 255]]"
        assert ex.max() <= 255, "Should be in [[0, 255]]"

        assert 0 <= prediction.min(), "Should be in [[0, 255]]"
        assert prediction.max() <= 255, "Should be in [[0, 255]]"

        # As examples are 511 × 512 × 512
        # We crop then to have the same shape on different axis
        first, last = 0, 511
        ex = ex[first:last, first:last, first:last]

        labels = (prediction[first:last, first:last, first:last] / 255) > \
                 args.prediction_level

        labels = np.array(labels, dtype=np.uint8)

        out_file = os.path.join(args.out_folder, ex_f.replace(".tif",
                                                              "_full.h5"))

        hf = h5py.File(out_file, "w")
        hf.create_dataset(name="raw", data=ex, chunks=True)
        hf.create_dataset(name="labels", data=labels, chunks=True)
        hf.close()

        print(f"Saved the example in {out_file}")
