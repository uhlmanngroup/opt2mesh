import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("example",
                        help="HDF5 file of one example (512 × 512 × 512 × n_classes) typically")
    parser.add_argument("ground_truth",
                        help="HDF5 file of labels of one (512 × 512 × 512 × 1), last channel in {0, …, n_classes}")
    parser.add_argument("out_folder",
                        help="Name of the output folder for slices")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Proportions of labels to consider a slice to be manually labelled")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

