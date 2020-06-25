import logging
import os
import time

import igl
import morphsnakes as ms
import numpy as np
import pymesh
from guppy import hpy
from skimage import io
from skimage.measure import marching_cubes

from settings import OUT_FOLDER

logging.basicConfig(level=logging.DEBUG)


def canonical_representation(v: np.array, f: np.array):
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


def tif2morphsnakes_halves(tif_file, iterations=150, smoothing=1,
                           lambda1=1, lambda2=2):
    """
    Create morphsnakes surfaces on the 6 halves of the cubes.to have the
    This allow to run the algoritm on the full 512³ resolution.

    Halves can be then merged together

    :param tif_file: path to the TIF stack to process
    :param iterations: number of iterations for the MCV algorithm
    :param smoothing: number of smoothing to perform for the MCV algorithm
    :param lambda1: weight parameter for the outer region for the MVA algorithm
    :param lambda2: weight parameter for the inner region for the MVA algorithm
    """

    h = hpy()
    print("Before loading the data")
    print(h.heap())

    # TODO: the half_size index should adapt to the data shape
    # the cube has a size of (511,512,512)
    half_size_index = 256

    for suffix in ["x_front", "x_back", "y_front", "y_back", "z_front", "z_back"]:

        logging.debug(f" → Loading the data for half {suffix}")

        # This is ugly, I haven't found something better
        if suffix == "x_front":
            opt_data = io.imread(tif_file)[:half_size_index, :, :]
        elif suffix == "x_back":
            opt_data = io.imread(tif_file)[half_size_index:, :, :]
        elif suffix == "y_front":
            opt_data = io.imread(tif_file)[:, :half_size_index, :]
        elif suffix == "y_back":
            opt_data = io.imread(tif_file)[:, half_size_index:, :]
        elif suffix == "z_front":
            opt_data = io.imread(tif_file)[:, :, :half_size_index]
        elif suffix == "z_back":
            opt_data = io.imread(tif_file)[:, :, half_size_index:]
        else:
            raise RuntimeError(f"{suffix} is a wrong suffix")

        logging.debug(f" → Loaded half {suffix}")
        print(h.heap())

        # Initialization of the level-set.
        init_ls = ms.circle_level_set(opt_data.shape)

        # Morphological Chan-Vese (or ACWE)
        logging.debug(f" → Loaded half {suffix}")

        logging.debug(f" → Running Morphological Chan Vese on {suffix}")

        start = time.time()

        half_surface = ms.morphological_chan_vese(opt_data,
                                                  init_level_set=init_ls,
                                                  iterations=iterations,
                                                  smoothing=smoothing,
                                                  lambda1=lambda1,
                                                  lambda2=lambda2)

        end = time.time()
        logging.debug(f" → Done Morphological Chan Vese on {suffix} in {(end - start) / 1000}s")

        half_surface_file = tif_file.replace(".tif", f"_{suffix}.tif")

        logging.debug(f" → Saving half {suffix} in: {half_surface_file}")
        io.imsave(half_surface_file, half_surface)


def morphsnakes_halves2surface(tif_file, save_reshaped_halves=True):
    """
    Merge different meshes together

    :param tif_file: path to the TIF stack to process
    :param save_reshaped_halves: save temporary halves
    :return:
    """

    x_front = io.imread(tif_file.replace(".tif", "_x_front.tif"))
    x_back = io.imread(tif_file.replace(".tif", "_x_back.tif"))

    logging.debug(f"x_front.shape         : {x_front.shape}")
    logging.debug(f"x_back.shape          : {x_back.shape}")

    # TODO: make this general                      v
    # the cube has a size of (511,512,512)
    x_front_reshaped = np.concatenate((x_front, np.zeros((255, 512, 512), dtype='int8')), axis=0)
    x_back_reshaped = np.concatenate((np.zeros((256, 512, 512), dtype='int8'), x_back), axis=0)

    logging.debug(f"x_front_reshaped.shape: {x_front_reshaped.shape}")
    logging.debug(f"x_back_reshaped.shape : {x_back_reshaped.shape}")

    y_front = io.imread(tif_file.replace(".tif", "_y_front.tif"))
    y_back = io.imread(tif_file.replace(".tif", "_y_back.tif"))

    logging.debug(f"y_front.shape         : {y_front.shape}")
    logging.debug(f"y_back.shape          : {y_back.shape}")

    y_front_reshaped = np.concatenate((y_front, np.zeros(y_front.shape, dtype='int8')), axis=1)
    y_back_reshaped = np.concatenate((np.zeros(y_back.shape, dtype='int8'), y_back), axis=1)

    logging.debug(f"y_front_reshaped.shape: {y_front_reshaped.shape}", )
    logging.debug(f"y_back_reshaped.shape : {y_front_reshaped.shape}")

    z_front = io.imread(tif_file.replace(".tif", "_z_front.tif"))
    z_back = io.imread(tif_file.replace(".tif", "_z_back.tif"))

    logging.debug(f"z_front.shape         : {z_front.shape}")
    logging.debug(f"z_back.shape          : {z_back.shape}")

    z_front_reshaped = np.concatenate((z_front, np.zeros(z_front.shape, dtype='uint8')), axis=2)
    z_back_reshaped = np.concatenate((np.zeros(z_back.shape, dtype='uint8'), z_back), axis=2)

    logging.debug(f"z_front_reshaped.shape: {z_front_reshaped.shape}")
    logging.debug(f"z_back_reshaped.shape : {z_back_reshaped.shape}")

    # The full segmentation surface
    full_surface = (x_front_reshaped + x_back_reshaped + y_back_reshaped
            + y_front_reshaped + z_back_reshaped + z_front_reshaped).clip(0, 1)

    if save_reshaped_halves:
        io.imsave(tif_file.replace(".tif", "_x_front_reshaped.tif"), x_front_reshaped)
        io.imsave(tif_file.replace(".tif", "_x_back_reshaped.tif"), x_back_reshaped)

        io.imsave(tif_file.replace(".tif", "_y_front_reshaped.tif"), y_front_reshaped)
        io.imsave(tif_file.replace(".tif", "_y_back_reshaped.tif"), y_back_reshaped)

        io.imsave(tif_file.replace(".tif", "_z_front_reshaped.tif"), z_front_reshaped)
        io.imsave(tif_file.replace(".tif", "_z_back_reshaped.tif"), z_back_reshaped)

    return full_surface


def clean_mesh(v, f):
    """
    Quick mesh cleaning.

    :param v: array of vertices
    :param f: array of faces
    :return:
    """
    logging.info(f" → Input mesh")
    logging.info(f"   → Vertices: {len(v)}")
    logging.info(f"   → Faces: {len(f)}")

    logging.info(f" → Removing isolated vertices")
    v, f, info = pymesh.remove_isolated_vertices_raw(v, f)
    logging.info(f"   → Num vertex removed: {info['num_vertex_removed']}")

    logging.info(f" → Removing duplicated vertices")
    v, f, info = pymesh.remove_duplicated_vertices_raw(v, f)
    logging.info(f"   → Num vertex merged: {info['num_vertex_merged']}")

    logging.info("f → Output mesh")
    logging.info(f"   → Vertices: {len(v)}")
    logging.info(f"   → Faces: {len(f)}")
    logging.info('')

    return v, f


def save_connected_components(tif_file, v, f):
    """
    Find and save connected components of a given mesh

    :param tif_file: path to the original TIF stack to process
    :param v: array of vertices
    :param f: array of faces
    :return:
    """

    mesh = pymesh.meshio.form_mesh(v, f)

    logging.info(f" → Splitting mesh in connected components")
    meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')
    meshes = sorted(meshes, key=lambda x: x.faces, reverse=True)
    logging.info(f"    → {len(meshes)} connected components")

    clean_mesh_file = tif_file.replace(".tif", "_cleaned_mesh.stl")

    for i, m in enumerate(meshes):
        vi = m.vertices
        fi = m.faces
        cc_mesh_file = clean_mesh_file.replace(".stl", f"_{i}.stl")

        logging.info(f" → Saving connected components in: {cc_mesh_file}")
        logging.info(f"   → Vertices: {len(vi)}")
        logging.info(f"   → Faces: {len(fi)}")
        logging.info('')
        igl.write_triangle_mesh(cc_mesh_file, vi, fi)


if __name__ == "__main__":
    data_folder = os.path.join(OUT_FOLDER, "morph_contours")

    tif_file = os.path.join(data_folder, "MNS_M897_115.tif")

    logging.debug(f" → Input file: {tif_file}")

    logging.debug(f" → Starting Morphological Chan Vese on halves")
    tif2morphsnakes_halves(tif_file)
    logging.debug(f" → Done Morphological Chan Vese on halves")

    full = morphsnakes_halves2surface(tif_file, save_reshaped_halves=True)
    io.imsave(tif_file.replace(".tif", "_surface.tif"), full)

    v, f, normals, values = marching_cubes(full,
                                           level=0.999,
                                           spacing=(1.0, 1.0, 1.0),
                                           gradient_direction='descent',
                                           step_size=1,
                                           allow_degenerate=True,
                                           mask=None)

    raw_mesh_file = tif_file.replace(".tif", "_raw_mesh.stl")
    igl.write_triangle_mesh(raw_mesh_file, v, f)

    v, f = clean_mesh(v, f)

    clean_mesh_file = tif_file.replace(".tif", "_cleaned_mesh.stl")

    logging.info(f" → Saving clean mesh in: {clean_mesh_file}")
    igl.write_triangle_mesh(clean_mesh_file, v, f)

    logging.info(f" → Connected components: {clean_mesh_file}")
    save_connected_components(tif_file, v, f)
