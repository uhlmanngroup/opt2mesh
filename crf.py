import argparse
import logging
import os

import numpy as np
from denseCRF3D import densecrf3d
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional Random Field Segmentation")

    # Argument
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run")

    # CRF3D arguments
    parser.add_argument("--max_iter", help="Maximum number of iterations to use",
                        type=int, default=2)

    parser.add_argument("--std_pos", help="Standard deviation",
                        type=float, default=3.0)
    parser.add_argument("--weight_pos", help="Weight",
                        type=float, default=10.0)

    parser.add_argument("--std_bilat", help="Standard deviation (bileteral)",
                        type=float, default=3.0)
    parser.add_argument("--weight_bilat", help="Weight (bilateral)",
                        type=float, default=15.0)

    return parser.parse_args()


def main():
    args = parse_args()

    tif_stack_file = args.in_tif
    opt_data = io.imread(tif_stack_file)

    os.makedirs(args.out_folder, exist_ok=True)

    # Convert a path like '/path/to/file.name.ext' to 'file.name'
    basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    file_basename = os.path.join(args.out_folder, basename)

    # TODO: cropped on the embryo here to be able to run on 8GioB
    # unitaries = np.asarray([opt_data[100:450, 200:400, :]])
    unitaries = np.asarray([opt_data])
    unitaries = np.transpose(unitaries, [1, 2, 3, 0])

    occupancy_neg_log_probs = - np.log(unitaries.astype("float32") / 255.0)

    dense_crf_params = {
        'MaxIterations': args.max_iter,
        'PosW': args.weight_pos,
        'PosRStd': args.std_pos,
        'PosCStd': args.std_pos,
        'PosZStd': args.std_pos,
        'BilateralW': args.weight_bilat,
        'BilateralRStd': args.std_bilat,
        'BilateralCStd': args.std_bilat,
        'BilateralZStd': args.std_bilat,
        'ModalityNum': 1,
        'BilateralModsStds': (5.0,)
    }

    # Number of channels: we only have one here

    logging.info(f"Running CRF3D.densecrf3D on "
                 f"cropped version of shape {unitaries.shape}")

    labels = densecrf3d(unitaries, occupancy_neg_log_probs, dense_crf_params)

    print(labels.shape, labels.dtype, )
    out_file = file_basename + f"_crf_surface" \
                               f"_{args.max_iter}" \
                               f"_{args.weight_pos}" \
                               f"_{args.std_pos}" \
                               f"_{args.weight_bilat}" \
                               f"_{args.std_bilat}.tif"
    logging.info(f"Saving segmentation in {out_file}")
    io.imsave(out_file, labels)


if __name__ == "__main__":
    main()
