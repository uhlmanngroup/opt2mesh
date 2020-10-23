#! /usr/bin/env python
import argparse
import logging
import os
import sys
import uuid

import yaml
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        prog="opt2mesh",
        description="Extract a mesh from a OPT scan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Argument
    parser.add_argument(
        "in_tif", help="Input OPT scan as tif stack (3D image)"
    )
    parser.add_argument(
        "out_folder", help="General output folder for this run"
    )
    parser.add_argument(
        "--method",
        help="Surface extraction method",
        choices=[
            "acwe",
            "gac",
            "autocontext",
            "autocontext_acwe",
            "2d_unet",
            "3d_unet",
            "direct",
        ],
        default="3d_unet",
    )

    # General settings
    parser.add_argument(
        "--save_temp", help="Save temporary results", action="store_true"
    )
    parser.add_argument(
        "--segment_occupancy_map",
        help="Segment the occupancy map",
        action="store_true",
    )
    parser.add_argument(
        "--save_occupancy_map",
        help="Save the occupancy map",
        action="store_true",
    )
    parser.add_argument(
        "--align_mesh",
        help="Align the mesh on the original OPT scan orientation",
        action="store_true",
    )

    parser.add_argument(
        "--timing", help="Print timing info", action="store_true"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=0,
        help="Number of jobs to use for parallel execution",
    )

    # Active contour general parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="ACWE & GAC: number of iterations",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help="ACWE & GAC: number of smoothing iteration (µ)",
    )

    # Geodesic active contour parameters
    parser.add_argument(
        "--threshold",
        help="GAC: number of smoothing iteration (µ)",
        default="auto",
    )
    parser.add_argument("--balloon", default=-1, help="GAC: ballon force")
    parser.add_argument(
        "--alpha",
        type=int,
        default=1000,
        help="GAC: inverse gradient transform alpha",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5,
        help="GAC: inverse gradient transform sigma",
    )

    # Active contour without edges Morphosnakes parameters
    parser.add_argument(
        "--on_halves",
        help="Adapt pipeline to be run the processing on "
        "halves instead on the full input tif stack",
        action="store_true",
    )
    parser.add_argument(
        "--on_slices",
        help="Adapt pipeline to be run the processing on "
        "slices instead on the full input tif stack",
        action="store_true",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=3,
        help="ACWE: weight parameter for the outer region",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=1,
        help="ACWE: weight parameter for the inner region",
    )

    # Auto-context segmentation parameters
    parser.add_argument(
        "--autocontext",
        type=str,
        help="Autocontext: path to the Ilastik project",
    )
    parser.add_argument(
        "--use_probabilities",
        help="Autocontext: use probabilities instead of"
        "segmentation for the occupancy map",
        action="store_true",
    )

    # UNet prediction parameters
    parser.add_argument(
        "--pytorch_model",
        default=None,
        metavar="FILE",
        help="UNet: Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="UNet (2D): Scale factor for the input images",
        default=0.5,
    )
    parser.add_argument(
        "--bilinear",
        help="UNet (2D): Use bilinear upsampling instead of Up Convolution",
        action="store_true",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="UNet (3D): Path to the YAML config file",
        default=None,
    )
    parser.add_argument(
        "--patch_halo",
        type=int,
        help="UNet (3D): Halo to remove from patch (one dimension)",
        default=None,
    )
    parser.add_argument(
        "--stride_shape",
        type=int,
        help="UNet (3D): Stride for the prediction (one dimension)",
        default=None,
    )
    parser.add_argument(
        "--f_maps",
        type=int,
        help="UNet (3D): Feature maps scale factor (one dimension)",
        default=None,
    )

    # Marching cubes parameters
    parser.add_argument(
        "--level",
        type=float,
        default=0.5,
        help="Marching Cubes: isolevel of the surface for marching cube",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Marching Cubes: spacing between voxels for marching cube",
    )
    parser.add_argument(
        "--gradient_direction",
        type=str,
        help="Marching Cubes: spacing between voxels",
        default="descent",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Marching Cubes: step size for marching cube",
    )

    def int_or_str(value):
        try:
            return int(value)
        except ValueError:
            return value

    # Mesh simplification parameters
    parser.add_argument(
        "--detail",
        help="Mesh decimation: Level of detail to preserve. "
        "Can be the target number of faces or a string in "
        "['low', 'normal', 'high', 'original']",
        default=3000,
        type=int_or_str,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # We first get information about the context of execution

    # Env variables set by LSF:
    # https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_config_ref/lsf_envars_job_exec.html
    job_id = os.getenv("LSB_JOBID", str(uuid.uuid1())[:8])
    job_batch_name = os.getenv("LSB_JOBNAME", "unknown_job_batch_name")
    now_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    job_out_folder = os.path.join(
        args.out_folder, str(job_id) + "_" + now_string
    )
    code_dir = os.path.abspath(
        os.path.join(__file__, os.pardir, os.pardir, os.pardir)
    )
    git_log_command = (
        f"git --git-dir={code_dir}/.git --work-tree={code_dir} log"
    )
    last_commit_message = os.popen(f"{git_log_command} -1").read().strip()
    last_commit = (
        os.popen(f'{git_log_command} -1 --pretty="%h"').read().strip()
    )
    cli_call = " ".join(sys.argv)

    os.makedirs(job_out_folder)
    logfile = os.path.join(job_out_folder, f"{job_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(),
        ],
    )

    # Keep in this order:
    #     job_id: 39492343
    #     job_batch_name: embryos
    #     input_file: …
    #     out_folder: …
    #     git_commit: …
    #     vsc_context: …
    #     cli_call: …
    #     arguments:
    #      - …
    #      - …
    #      - …

    context = dict()
    context["job_id"] = job_id
    context["job_batch_name"] = job_batch_name
    context["input_file"] = args.in_tif
    context["out_folder"] = job_out_folder
    context["git_commit"] = last_commit
    context["vsc_context"] = last_commit_message
    context["cli_call"] = cli_call

    mesh_info_file = os.path.join(job_out_folder, f"{job_id}_context.yml")

    with open(mesh_info_file, "w") as fp:
        yaml.dump(context, fp, sort_keys=False)

    # Logging the context
    logging.info("CLI call:")
    logging.info(cli_call)

    logging.info("LSF environnement:")
    # Env variables set by LSF:
    # https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_config_ref/lsf_envars_job_exec.html
    for lsf_env_var in [
        "LSB_BIND_CPU_LIST",
        "LSB_HOSTS",
        "LSB_QUEUE",
        "LSB_JOBNAME",
        "LSB_JOB_CWD",
    ]:
        logging.info(f"  {lsf_env_var}: {os.getenv(lsf_env_var, 'Not set')}")

    # Adapt the number of jobs if not set
    if args.n_jobs <= 0:
        # LSB_BIND_CPU_LIST of the form: "13,23,24,71"
        args.n_jobs = len(os.getenv("LSB_BIND_CPU_LIST", "").split(","))
        logging.info(
            f"Adapting the number of jobs to number of available CPU {args.n_jobs}"
        )

    logging.info("VCS context:")
    logging.info(last_commit_message)

    ###

    if args.method.lower() == "gac":
        from pipeline.active_contours import GACPipeline

        opt2mesh_pipeline = GACPipeline(
            # GAC specifics
            iterations=args.iterations,
            on_slices=args.on_slices,
            n_jobs=args.n_jobs,
            smoothing=args.smoothing,
            threshold=args.threshold,
            balloon=args.balloon,
            alpha=args.alpha,
            sigma=args.sigma,
            ###
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "acwe":
        from pipeline.active_contours import ACWEPipeline

        opt2mesh_pipeline = ACWEPipeline(
            # ACWE specifics
            iterations=args.iterations,
            on_slices=args.on_slices,
            n_jobs=args.n_jobs,
            on_halves=args.on_halves,
            smoothing=args.smoothing,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            #
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "autocontext":
        from pipeline.ilastik import AutoContextPipeline

        opt2mesh_pipeline = AutoContextPipeline(
            # AutoContextSpecific
            project=args.autocontext,
            use_probabilities=args.use_probabilities,
            ###
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "autocontext_acwe":
        from pipeline.ilastik import AutoContextACWEPipeline

        opt2mesh_pipeline = AutoContextACWEPipeline(
            # AutoContextSpecific
            project=args.autocontext,
            # ACWE specifics
            smoothing=args.smoothing,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            iterations=args.iterations,
            ###
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "2d_unet":
        from pipeline.unet import UNetPipeline

        opt2mesh_pipeline = UNetPipeline(
            # UNet specifics
            model_file=args.pytorch_model,
            scale_factor=args.scale,
            bilinear=args.bilinear,
            ###
            level=args.level,
            ###
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "3d_unet":
        from pipeline.unet import UNet3DPipeline

        opt2mesh_pipeline = UNet3DPipeline(
            # UNet specifics
            model_file=args.pytorch_model,
            config_file=args.config_file,
            patch_halo=args.patch_halo,
            stride_shape=args.stride_shape,
            f_maps=args.f_maps,
            ###
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    elif args.method.lower() == "direct":
        from pipeline.base import DirectMeshingPipeline

        opt2mesh_pipeline = DirectMeshingPipeline(
            level=args.level,
            spacing=args.spacing,
            gradient_direction=args.gradient_direction,
            step_size=args.step_size,
            detail=args.detail,
            save_temp=args.save_temp,
            segment_occupancy_map=args.segment_occupancy_map,
            save_occupancy_map=args.save_occupancy_map,
            align_mesh=args.align_mesh,
        )
    else:
        raise RuntimeError(f"Method {args.method} is not recognised")

    logging.info(f"Starting pipeline {opt2mesh_pipeline.__class__.__name__}")
    logging.info(f"  Input TIF stack: {args.in_tif}")
    logging.info(f"  Out folder: {job_out_folder}")
    _, _, mesh_info = opt2mesh_pipeline.run(
        tif_stack_file=args.in_tif, out_folder=job_out_folder
    )
    logging.info(f"End of pipeline {opt2mesh_pipeline.__class__.__name__}")

    logging.info("Mesh correctness:")
    for k, v in mesh_info["mesh_correctness"].items():
        key = k.replace("_", " ").capitalize()
        logging.info(f"   {key}: {v}")
    logging.info("Mesh statistics:")
    for k, v in mesh_info["mesh_correctness"].items():
        key = k.replace("_", " ").capitalize()
        logging.info(f"   {key}: {v}")

    mesh_info_file = os.path.join(job_out_folder, f"{job_id}_mesh_quality.yml")
    logging.info(f"Saving mesh information in {mesh_info_file}")
    with open(mesh_info_file, "w") as fp:
        yaml.dump(mesh_info, fp, sort_keys=False)


if __name__ == "__main__":
    main()
