#! /usr/bin/env python
import argparse
import sys
from datetime import datetime
import logging
import os
import time

import igl
import morphsnakes as ms
import numpy as np
import pymesh
from guppy import hpy
from numpy.linalg import norm
from skimage import io, measure

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


class TIF2Mesh:
    """
    Pipeline to convert TIF stacks of images to a STL mesh file.

    """

    def __init__(self, on_halves=True,
                 save_temp=True,
                 iterations=50,
                 smoothing=1,
                 lambda1=1,
                 lambda2=2,
                 level=0.999,
                 spacing=1,
                 gradient_direction="descent",
                 step_size=1,
                 timing=True,
                 detail="high"):
        self.on_halves: bool = on_halves
        self.save_temp: bool = save_temp
        self.iterations: int = iterations
        self.smoothing: int = smoothing
        self.lambda1: int = lambda1
        self.lambda2: int = lambda2
        self.level: float = level
        self.spacing: int = spacing
        self.gradient_direction: str = gradient_direction
        self.step_size: int = step_size
        self.timing: bool = timing
        self.detail: str = detail

    def run(self, tif_stack_file, out_folder):
        os.makedirs(out_folder, exist_ok=True)

        # path/to/file.name.ext file.name
        basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

        base_out_file = os.path.join(out_folder, basename)

        logging.info(f"Input file: {tif_stack_file}")

        if self.on_halves:
            logging.info(f"Starting Morphological Chan Vese on halves")
            self._tif2morphsnakes_halves(tif_stack_file, base_out_file)
            logging.info(f"Done Morphological Chan Vese on halves")
            full_surface = self._morphsnakes_halves2surface(base_out_file)
        else:
            logging.info(f"Starting Morphological Chan Vese on the full dataset")
            logging.info(f"Loading full data")
            opt_data = io.imread(tif_stack_file)
            logging.info(f"Loaded full data")

            # Initialization of the level-set.
            init_ls = ms.circle_level_set(opt_data.shape)

            logging.info(f"Running Morphological Chan Vese on full")

            start = time.time()

            full_surface = ms.morphological_chan_vese(opt_data,
                                                      init_level_set=init_ls,
                                                      iterations=self.iterations,
                                                      smoothing=self.smoothing,
                                                      lambda1=self.lambda1,
                                                      lambda2=self.lambda2)

            end = time.time()
            logging.info(f"Done Morphological Chan Vese on full in {(end - start) / 1000}s")
            del opt_data, init_ls

        io.imsave(base_out_file + "_surface.tif", full_surface)

        v, f, normals, values = measure.marching_cubes(full_surface,
                                                       level=self.level,
                                                       spacing=(self.spacing, self.spacing, self.spacing),
                                                       gradient_direction=self.gradient_direction,
                                                       step_size=self.step_size,
                                                       # we enforce non-degeneration
                                                       allow_degenerate=True,
                                                       mask=None)

        raw_mesh_file = base_out_file + "_raw_mesh.stl"
        igl.write_triangle_mesh(raw_mesh_file, v, f)

        v, f = self.clean_mesh(v, f)

        clean_mesh_file = base_out_file + "_cleaned_mesh.stl"

        logging.info(f"Saving clean mesh in: {clean_mesh_file}")
        igl.write_triangle_mesh(clean_mesh_file, v, f)

        mesh = pymesh.meshio.form_mesh(v, f)

        logging.info(f"Splitting mesh in connected components")
        meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')
        #meshes = sorted(meshes, key=lambda x: x.faces, reverse=True)
        logging.info(f"  {len(meshes)} connected components")

        for i, m in enumerate(meshes):
            vi = m.vertices
            fi = m.faces
            cc_mesh_file = clean_mesh_file.replace(".stl", f"_{i}.stl")

            logging.info(f"{i + 1}th connected component ")
            logging.info(f" Vertices: {len(vi)}")
            logging.info(f" Faces: {len(fi)}")
            logging.info('')
            if self.save_temp:
                logging.info(f"Saving connected components #{i}: {cc_mesh_file}")
                igl.write_triangle_mesh(cc_mesh_file, vi, fi)

        # Taking the main mesh
        mesh_to_simplify = meshes[0]

        logging.info(f"Final mesh simplification")
        final_output_mesh = self._mesh_simplification(mesh_to_simplify)

        final_mesh_file = base_out_file + "_cleaned_mesh.stl"
        logging.info(f"Saving final simplified mesh in: {final_mesh_file}")
        pymesh.meshio.save_mesh(final_mesh_file, final_output_mesh)
        logging.info(f"Saved final simplified mesh !")
        logging.info("Pipeline done!")

    def _tif2morphsnakes_halves(self, tif_stack_file, base_out_file):
        """
        Create morphsnakes surfaces on the 6 halves of the cubes.to have the
        This allow to run the algoritm on the full 512³ resolution.

        Halves can be then merged together

        :param tif_stack_file: path to the TIF stack to process
        :param base_out_file:
        """

        h = hpy()
        logging.info("Before loading the data")
        logging.info(str(h.heap()))

        # TODO: the half_size index should adapt to the data shape
        # the cube has a size of (511,512,512)
        half_size_index = 256

        for suffix in ["x_front", "x_back", "y_front", "y_back", "z_front", "z_back"]:

            logging.info(f"Loading the data for half {suffix}")

            # This is ugly, I haven't found something better
            if suffix == "x_front":
                opt_data = io.imread(tif_stack_file)[:half_size_index, :, :]
            elif suffix == "x_back":
                opt_data = io.imread(tif_stack_file)[half_size_index:, :, :]
            elif suffix == "y_front":
                opt_data = io.imread(tif_stack_file)[:, :half_size_index, :]
            elif suffix == "y_back":
                opt_data = io.imread(tif_stack_file)[:, half_size_index:, :]
            elif suffix == "z_front":
                opt_data = io.imread(tif_stack_file)[:, :, :half_size_index]
            elif suffix == "z_back":
                opt_data = io.imread(tif_stack_file)[:, :, half_size_index:]
            else:
                raise RuntimeError(f"{suffix} is a wrong suffix")

            logging.info(f"Loaded half {suffix}")
            logging.info(str(h.heap()))

            # Initialization of the level-set.
            init_ls = ms.circle_level_set(opt_data.shape)

            # Morphological Chan-Vese (or ACWE)
            logging.info(f"Loaded half {suffix}")

            logging.info(f"Running Morphological Chan Vese on {suffix}")

            start = time.time()

            half_surface = ms.morphological_chan_vese(opt_data,
                                                      init_level_set=init_ls,
                                                      iterations=self.iterations,
                                                      smoothing=self.smoothing,
                                                      lambda1=self.lambda1,
                                                      lambda2=self.lambda2)

            end = time.time()
            logging.info(f"Done Morphological Chan Vese on {suffix} in {(end - start) / 1000}s")

            half_surface_file = base_out_file + f"_{suffix}.tif"

            logging.info(f"Saving half {suffix} in: {half_surface_file}")
            io.imsave(half_surface_file, half_surface)

    def _morphsnakes_halves2surface(self, base_out_file):
        """
        Merge different meshes together

        :param base_out_file: path to the TIF stack to process
        :return:
        """

        x_front = io.imread(base_out_file + "_x_front.tif")
        x_back = io.imread(base_out_file + "_x_back.tif")

        logging.info(f"x_front.shape         : {x_front.shape}")
        logging.info(f"x_back.shape          : {x_back.shape}")

        # TODO: make this general                      v
        # the cube has a size of (511,512,512)
        x_front_reshaped = np.concatenate((x_front, np.zeros((255, 512, 512), dtype='int8')), axis=0)
        x_back_reshaped = np.concatenate((np.zeros((256, 512, 512), dtype='int8'), x_back), axis=0)

        logging.info(f"x_front_reshaped.shape: {x_front_reshaped.shape}")
        logging.info(f"x_back_reshaped.shape : {x_back_reshaped.shape}")

        y_front = io.imread(base_out_file + "_y_front.tif")
        y_back = io.imread(base_out_file + "_y_back.tif")

        logging.info(f"y_front.shape         : {y_front.shape}")
        logging.info(f"y_back.shape          : {y_back.shape}")

        y_front_reshaped = np.concatenate((y_front, np.zeros(y_front.shape, dtype='int8')), axis=1)
        y_back_reshaped = np.concatenate((np.zeros(y_back.shape, dtype='int8'), y_back), axis=1)

        logging.info(f"y_front_reshaped.shape: {y_front_reshaped.shape}", )
        logging.info(f"y_back_reshaped.shape : {y_front_reshaped.shape}")

        z_front = io.imread(base_out_file + "_z_front.tif")
        z_back = io.imread(base_out_file + "_z_back.tif")

        logging.info(f"z_front.shape         : {z_front.shape}")
        logging.info(f"z_back.shape          : {z_back.shape}")

        z_front_reshaped = np.concatenate((z_front, np.zeros(z_front.shape, dtype='uint8')), axis=2)
        z_back_reshaped = np.concatenate((np.zeros(z_back.shape, dtype='uint8'), z_back), axis=2)

        logging.info(f"z_front_reshaped.shape: {z_front_reshaped.shape}")
        logging.info(f"z_back_reshaped.shape : {z_back_reshaped.shape}")

        # The full segmentation surface
        full_surface = (x_front_reshaped + x_back_reshaped + y_back_reshaped
                        + y_front_reshaped + z_back_reshaped + z_front_reshaped).clip(0, 1)

        if self.save_temp:
            io.imsave(base_out_file + "_x_front_reshaped.tif", x_front_reshaped)
            io.imsave(base_out_file + "_x_back_reshaped.tif", x_back_reshaped)

            io.imsave(base_out_file + "_y_front_reshaped.tif", y_front_reshaped)
            io.imsave(base_out_file + "_y_back_reshaped.tif", y_back_reshaped)

            io.imsave(base_out_file + "_z_front_reshaped.tif", z_front_reshaped)
            io.imsave(base_out_file + "_z_back_reshaped.tif", z_back_reshaped)

        return full_surface

    @staticmethod
    def clean_mesh(v, f):
        """
        Quick mesh cleaning.

        :param v: array of vertices
        :param f: array of faces
        :return:
        """
        logging.info(f"Input mesh")
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces: {len(f)}")

        logging.info(f"Removing isolated vertices")
        v, f, info = pymesh.remove_isolated_vertices_raw(v, f)
        logging.info(f"  Num vertex removed: {info['num_vertex_removed']}")

        logging.info(f"Removing duplicated vertices")
        v, f, info = pymesh.remove_duplicated_vertices_raw(v, f)
        logging.info(f"  Num vertex merged: {info['num_vertex_merged']}")

        logging.info("f Output mesh")
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces: {len(f)}")
        logging.info('')

        return v, f

    def _mesh_simplification(self, mesh):
        """
        Remesh the input mesh to remove degeneracies and improve triangle quality.

        Taken and adapted from:
        https://github.com/PyMesh/PyMesh/blob/master/scripts/fix_mesh.py
        """
        bbox_min, bbox_max = mesh.bbox
        diag_len = norm(bbox_max - bbox_min)
        if self.detail == "normal":
            target_len = diag_len * 5e-3
        elif self.detail == "high":
            target_len = diag_len * 2.5e-3
        elif self.detail == "low":
            target_len = diag_len * 1e-2
        logging.info("Target resolution: {} mm".format(target_len))

        count = 0
        mesh, __ = pymesh.remove_degenerated_triangles(mesh, num_iterations=100)
        mesh, __ = pymesh.split_long_edges(mesh, target_len)
        num_vertices = mesh.num_vertices
        while True:
            mesh, __ = pymesh.collapse_short_edges(mesh, abs_threshold=1e-6)
            mesh, __ = pymesh.collapse_short_edges(mesh, abs_threshold=target_len,
                                                   preserve_feature=True)
            mesh, __ = pymesh.remove_obtuse_triangles(mesh, max_angle=150.0,
                                                      max_iterations=100)
            if mesh.num_vertices == num_vertices:
                break

            num_vertices = mesh.num_vertices
            logging.info("# Number of vertices: {}".format(num_vertices))
            count += 1
            if count > 10:
                break

        mesh = pymesh.resolve_self_intersection(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh = pymesh.compute_outer_hull(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, max_angle=179.0,
                                                  max_iterations=5)
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')
        # meshes = sorted(meshes, key=lambda x: x.faces, reverse=True)

        # Once again, we take the first connected component
        final_mesh = meshes[0]

        return final_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline ")

    # Argument
    parser.add_argument("in_tif", help="Input tif stack (3D image)")
    parser.add_argument("out_folder", help="General output folder for this run",
                        default=os.path.join(OUT_FOLDER, "tif2mesh"))

    # Data wise
    parser.add_argument("--on_halves", help="Adapt pipeline to be run the processing on "
                                            "halves instead on the full input tif stack",
                        action="store_true")
    parser.add_argument("--save_temp", help="Save temporary results",
                        action="store_true")
    parser.add_argument("--timing", help="Print timing info", action="store_true")

    # Morphosnakes parameters
    parser.add_argument("--iterations", help="Morphosnakes: number of iterations", default=50)
    parser.add_argument("--smoothing", help="Morphosnakes: number of smoothing iteration (µ)", default=1)
    parser.add_argument("--lambda1", help="Morphosnakes: weight parameter for the outer region", default=1)
    parser.add_argument("--lambda2", help="Morphosnakes: weight parameter for the inner region", default=2)

    # Marching cubes parameters
    parser.add_argument("--level", help="Marching Cubes: isolevel of the surface for marching cube", default=0.999)
    parser.add_argument("--spacing", help="Marching Cubes: spacing between voxels for marching cube", default=1.0)
    parser.add_argument("--gradient_direction", help="Marching Cubes: spacing between voxels", default="descent")
    parser.add_argument("--step_size", help="Marching Cubes: step size for marching cube", default=1)

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
    out_folder = os.path.join(args.out_folder, now_string())

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
        logging.info("  %s: %r", arg, value)

    tif2mesh_pipeline = TIF2Mesh(on_halves=args.on_halves,
                                 save_temp=args.save_temp,
                                 iterations=args.iterations,
                                 smoothing=args.smoothing,
                                 lambda1=args.lambda1,
                                 lambda2=args.lambda2,
                                 level=args.level,
                                 spacing=args.spacing,
                                 gradient_direction=args.gradient_direction,
                                 step_size=args.step_size,
                                 timing=args.timing,
                                 detail=args.detail)

    logging.info(f"Starting TIF2Mesh pipeline")
    logging.info(f"  Input TIF stack: {args.in_tif}")
    logging.info(f"  Out folder: {out_folder}")
    tif2mesh_pipeline.run(tif_stack_file=args.in_tif, out_folder=out_folder)

    logging.info("End of TIF2Mesh pipeline")


if __name__ == "__main__":
    main()
