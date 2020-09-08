#! /usr/bin/env python

import h5py
import sys
from skimage import io

if __name__ == "__main__":
    data = io.imread(sys.argv[1])

    hf = h5py.File(sys.argv[1].replace(".tif", ".h5"), "w")
    hf.create_dataset(name="dataset", data=data, chunks=True)
    hf.close()
