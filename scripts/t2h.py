#! /usr/bin/env python

import h5py
import argparse
from skimage import io

__doc__ = "TIF to HDF5 converter"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_file", help="TIF stack")
    args = parser.parse_args()

    data = io.imread(args.tif_file)
    hf = h5py.File(args.tif_file.replace(".tif", ".h5"), "w")
    hf.create_dataset(name="dataset", data=data, chunks=True)
    hf.close()
