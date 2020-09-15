from skimage import io
import argparse
import numpy as np

__doc__ = "Binarise a tif stack based on a decision threshold" \
          "used to construct the ground truth."

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_file", help="TIF stack")
    parser.add_argument("threshold",
                        help="Value in [0, 1] to binarise image")

    args = parser.parse_args()

    data = io.imread(args.tif_file)
    data_out = np.array(data > args.threshold, dtype=np.uint8)
    io.imsave(args.tif_file.replace(".tif", "bin.tif"), data_out)

