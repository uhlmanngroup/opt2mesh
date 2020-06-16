import os
import numpy as np
import meshplot
import igl
import logging
import pymesh

logging.basicConfig(level=logging.INFO)


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

    in_data_folder = "../../data/Limbs/Late/Forelimb/MUT/"

    out_data_folder = in_data_folder.replace("data", "out")

    os.makedirs(out_data_folder, exist_ok=True)

    for file in sorted(os.listdir(in_data_folder)):
        infilepath = os.path.join(in_data_folder, file)

        logging.info(f" → Loading: {infilepath.split(os.sep)[-1]}")
        v, f = igl.read_triangle_mesh(infilepath)
        v, f = canonical_representation(v, f)
        logging.info(f"   → Vertices: {len(v)}")
        logging.info(f"   → Faces: {len(f)}")

        logging.info(f" → Removing isolated vertices")
        v, f, info = pymesh.remove_isolated_vertices_raw(v, f)
        logging.info(f"Num vertex removed: {info['num_vertex_removed']}")

        logging.info(f" → Removing duplicated vertices")
        v, f, info = pymesh.remove_duplicated_vertices_raw(v, f)
        logging.info(f"Num vertex merged: {info['num_vertex_merged']}")

        mesh = pymesh.meshio.form_mesh(v, f)

        logging.info(f" → Splitting mesh in connected components")
        meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')
        logging.info(f"    → {len(meshes)} connected components")

        outfilepath = os.path.join(out_data_folder, file)

        logging.info(f" → Saving results in: {outfilepath.split(os.sep)[-1]}")
        logging.info(f"   → Vertices: {len(v)}")
        logging.info(f"   → Faces: {len(f)}")
        logging.info('')

        igl.write_triangle_mesh(outfilepath, v, f)

        for i, m in enumerate(meshes):
            vi = m.vertices
            fi = m.faces
            outfilepath = os.path.join(out_data_folder, file).replace(".stl", f"_{i}.stl")

            logging.info(f" → Saving connected components results in: {outfilepath.split(os.sep)[-1]}")
            logging.info(f"   → Vertices: {len(vi)}")
            logging.info(f"   → Faces: {len(fi)}")
            logging.info('')
            igl.write_triangle_mesh(outfilepath, vi, fi)

