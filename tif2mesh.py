#! /usr/bin/env python
import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np

from pipeline import ACWEPipeline, GACPipeline
from settings import OUT_FOLDER


def _canonical_representation(v: np.array, f: np.array):
    """
    Convert a mesh to a canonical representation using
    PCA rotation matrix.

    :param v: input vertices array
    :param f: input vertices
    :return:
    """

    centered_v = v - np.mean(v, axis=0)

    # Getting the rotation matrix by diagonalizing the
    # covariance matrix of the centered protein coordinates
    cov_matrix = np.cov(centered_v.T)
    assert cov_matrix.shape == (3, 3)
    eigen_vals, rotation_mat = np.linalg.eig(cov_matrix)

    # Applying this rotation matrix on all points
    # Note : we should not transpose the rotation matrix (tested)
    logging.info("Rotation matrix:")
    logging.info(rotation_mat)
    v_rot = centered_v.dot(rotation_mat)

    return v_rot, f


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline ")

    # Argument
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run",
                        default=os.path.join(OUT_FOLDER, "tif2mesh"))
    parser.add_argument("--method", help="Surface extraction method",
                        choices=["acwe", "gac"], default="acwe")

    # Data wise
    parser.add_argument("--save_temp", help="Save temporary results",
                        action="store_true")
    parser.add_argument("--timing", help="Print timing info", action="store_true")

    # Active contour general parameters
    parser.add_argument("--iterations", type=int, help="ACWE & GAC: number of iterations", default=50)
    parser.add_argument("--smoothing", type=int, help="ACWE & GAC: number of smoothing iteration (µ)", default=1)

    # Geodesic active contour parameters
    parser.add_argument("--threshold", help="GAC: number of smoothing iteration (µ)", default="auto")
    parser.add_argument("--balloon", help="GAC: ballon force", default=1)
    parser.add_argument("--alpha", type=int, help="GAC: inverse gradient transform alpha", default=1000)
    parser.add_argument("--sigma", type=float, help="GAC: inverse gradient transform sigma", default=5)

    # Active contour without edges Morphosnakes parameters
    parser.add_argument("--on_halves", help="Adapt pipeline to be run the processing on "
                                            "halves instead on the full input tif stack",
                        action="store_true")
    parser.add_argument("--on_slices", help="Adapt pipeline to be run the processing on "
                                            "slices instead on the full input tif stack",
                        action="store_true")
    parser.add_argument("--lambda1", type=int, help="ACWE: weight parameter for the outer region", default=1)
    parser.add_argument("--lambda2", type=int, help="ACWE: weight parameter for the inner region", default=2)

    # Marching cubes parameters
    parser.add_argument("--level", type=float, help="Marching Cubes: isolevel of the surface for marching cube",
                        default=0.999)
    parser.add_argument("--spacing", type=float, help="Marching Cubes: spacing between voxels for marching cube",
                        default=1.0)
    parser.add_argument("--gradient_direction", type=str, help="Marching Cubes: spacing between voxels",
                        default="descent")
    parser.add_argument("--step_size", type=int, help="Marching Cubes: step size for marching cube", default=1)

    # Mesh simplification parameters
    parser.add_argument("--detail", help="Mesh simplification: Level of detail to preserve",
                        choices=["low", "normal", "high"], default="normal")

    return parser.parse_args()


def now_string():
    """
    Return a string of the current datetime.

    :return:
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def main():
    args = parse_args()

    # Env variables set by LSF:
    # https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_config_ref/lsf_envars_job_exec.html
    job_id = os.getenv('LSB_JOBID', 0)

    out_folder = os.path.join(args.out_folder, str(job_id) + "_" + now_string())

    os.makedirs(out_folder)

    logfile = os.path.join(out_folder, "run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    logging.info("CLI call:")
    logging.info("".join(sys.argv))

    logging.info("Arguments got ")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    logging.info("LSF environnement:")
    for env_var in ["LSB_BIND_CPU_LIST", "LSB_HOSTS", "LSB_QUEUE", "LSB_JOBNAME", "LSB_JOB_CWD"]:
        logging.info(f"  {env_var}: {os.getenv(env_var, 'Not set')}")

    if args.method.lower() == "gac":
        tif2mesh_pipeline = GACPipeline(iterations=args.iterations,
                                        level=args.level,
                                        spacing=args.spacing,
                                        gradient_direction=args.gradient_direction,
                                        step_size=args.step_size,
                                        timing=args.timing,
                                        detail=args.detail,
                                        save_temp=args.save_temp,
                                        on_slices=args.on_slices,
                                        # GAC specifics
                                        smoothing=args.smoothing,
                                        threshold=args.threshold,
                                        balloon=args.balloon,
                                        alpha=args.alpha,
                                        sigma=args.sigma)
    elif args.method.lower() == "acwe":
        tif2mesh_pipeline = ACWEPipeline(iterations=args.iterations,
                                         level=args.level,
                                         spacing=args.spacing,
                                         gradient_direction=args.gradient_direction,
                                         step_size=args.step_size,
                                         timing=args.timing,
                                         detail=args.detail,
                                         save_temp=args.save_temp,
                                         on_slices=args.on_slices,
                                         # ACWE specifics
                                         on_halves=args.on_halves,
                                         smoothing=args.smoothing,
                                         lambda1=args.lambda1,
                                         lambda2=args.lambda2)
    else:
        raise RuntimeError(f"Method {args.method} is not recongnised")

    logging.info(f"Starting TIF2Mesh pipeline")
    logging.info(f"  Input TIF stack: {args.in_tif}")
    logging.info(f"  Out folder: {out_folder}")
    tif2mesh_pipeline.run(tif_stack_file=args.in_tif, out_folder=out_folder)

    logging.info("End of TIF2Mesh pipeline")


if __name__ == "__main__":
    main()
