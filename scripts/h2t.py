#! /usr/bin/env python

import h5py
import argparse
import numpy as np
from skimage import io

__doc__ = "H5 to TIF converter"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("h5_file", help="TIF stack")
    args = parser.parse_args()

    hf = h5py.File(args.h5_file, "r")
    key = list(hf.keys())[0]
    data = np.array(hf[key], dtype=np.uint8)
    hf.close()

    io.imsave(args.h5_file.replace(".h5", ".tif"), data)
