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
    data = np.array(hf[key])
    hf.close()

    print(f"{args.h5_file} dtype: {data.dtype}")

    if data.dtype == np.uint8:
        print(f"Data is already {data.dtype}, do not convert")
        assert 0 < data.min() <= 255, "Int data must be between 0 and 255"
        assert 0 < data.max() <= 255, "Int data must be between 0 and 255"

    if data.dtype == np.float32 or data.dtype == np.float64:
        print(f"Data is {data.dtype}, do convert to np.uint8 values")
        assert 0 < data.min() <= 1, "Float data must be between 0 and 1"
        assert 0 < data.max() <= 1, "Float data must be between 0 and 1"
        data = np.array(data * 255, dtype=np.uint8)

    io.imsave(args.h5_file.replace(".h5", ".tif"), data)
