import os
import numpy as np
import meshplot
import igl
import logging

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


if __name__ == "__main__":

    # To be used when not in a notebook https://github.com/skoch9/meshplot/issues/11
    meshplot.offline()

    in_data_folder = "../../data/Limbs/Early/Forelimb/WT/"

    out_data_folder = in_data_folder.replace("data", "out")

    os.makedirs(out_data_folder, exist_ok=True)

    for file in os.listdir(in_data_folder):
        filepath = os.path.join(in_data_folder, file)

        v, f = igl.read_triangle_mesh(filepath)
        v_rot, f = canonical_representation(v, f)
        logging.info(f"Vertices: {len(v)}")
        logging.info(f"Faces: {len(f)}")

        outfile = file.replace(".stl", "_fixed.stl")
        outfilepath = os.path.join(out_data_folder, outfile)
        igl.write_triangle_mesh(outfilepath, v_rot, f)
