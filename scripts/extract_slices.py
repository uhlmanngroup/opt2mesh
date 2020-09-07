#! /usr/bin/env python

import argparse
import os

import h5py
import numpy as np
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("example",
                        help="HDF5 file of one example (512 × 512 × 512 × n_classes) typically")
    parser.add_argument("out_folder",
                        help="Name of the output folder for slices")
    parser.add_argument("--first", default=20, type=int,
                        help="The first slice to extract")
    parser.add_argument("--last", default=500, type=int,
                        help="The last slice to extract")
    parser.add_argument("--n_slices", default=5, type=int,
                        help="The last slice to extract")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if ".h5" in args.example:
        hf = h5py.File(args.example, "r")
        # Assuming the first key is the one to use
        key = list(hf.keys())[0]
        data = np.array(hf[key])
    elif ".tif" in args.example:
        data = io.imread(args.example)
    else:
        raise RuntimeError(f"Can't open {args.example}: must be h5 or tif")

    h, w, d = data.shape

    # /path/to/base.file.name.ext → base.file.name
    basefile_name = "".join(args.example.split(os.sep)[-1].split(".")[:-1])

    example_out_folder = os.path.join(args.out_folder, basefile_name)

    os.makedirs(example_out_folder, exist_ok=True)
    out_basefile_name = os.path.join(example_out_folder, basefile_name)

    for x in np.linspace(start=args.first,
                         stop=min(args.last, h),
                         num=args.n_slices,
                         dtype=int):
        file_name = out_basefile_name + f'_x_{x}.tif'
        slice = data[x, :, :]
        io.imsave(file_name, slice)

    for y in np.linspace(start=args.first,
                         stop=min(args.last, w),
                         num=args.n_slices,
                         dtype=int):
        file_name = out_basefile_name + f'_y_{y}.tif'
        slice = data[:, y, :]
        io.imsave(file_name, slice)

    for z in np.linspace(start=args.first,
                         stop=min(args.last, d),
                         num=args.n_slices,
                         dtype=int):
        file_name = out_basefile_name + f'_z_{z}.tif'
        slice = data[:, :, z]
        io.imsave(file_name, slice)