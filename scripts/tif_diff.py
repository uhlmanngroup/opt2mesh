#! /usr/bin/env python

import argparse
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

__doc__ = "Compute the difference of two TIF : C = A - B"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument(
        "A",
        help="First TIF volume / slice",
    )
    parser.add_argument(
        "B",
        help="Second TIF volume / slice",
    )

    parser.add_argument("C", help="Name of the difference.")

    args = parser.parse_args()

    A = io.imread(args.A)
    B = io.imread(args.B)

    for slice_index in [173, 258]:
        a = np.array(A[slice_index] / 255, dtype=np.float32)
        b = np.array(B[slice_index] / 255, dtype=np.float32)
        c = a - b

        plt.axis("off")
        plt.tight_layout()
        ax = plt.imshow(c, cmap="RdBu", vmin=-1.2, vmax=1.2)
        plt.colorbar(ax)
        filename = args.C + "_" + str(slice_index).zfill(4) + ".png"
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close("all")
