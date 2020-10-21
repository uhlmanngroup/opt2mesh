import logging
import os
import uuid
from abc import ABC, abstractmethod

import h5py
import igl
import numpy as np
import pymesh
import pymeshfix
from scipy.linalg import norm
from skimage import io, measure
from skimage.morphology import flood_fill


class OPT2MeshPipeline(ABC):
    """
    General pipeline to convert a OPT to a mesh.

    Input: TIF Stack
    Output: STL file

    This pipeline:
      Loads OPT scan
      Segment embryo
      Extracts mesh
      Correct and simplify mesh

    """

    def __init__(
        self,
        level=0.5,
        spacing=1,
        gradient_direction="descent",
        step_size=1,
        detail="high",
        save_temp=False,
        segment_occupancy_map=False,
        save_occupancy_map=False,
    ):
        self.level: float = level
        self.spacing: int = spacing
        self.gradient_direction: str = gradient_direction
        self.step_size: int = step_size
        self.detail: str = detail
        self.save_temp: bool = save_temp
        self.segment_occupancy_map: bool = segment_occupancy_map
        self.save_occupancy_map: bool = save_occupancy_map

    @abstractmethod
    def _extract_occupancy_map(self, tif_stack_file, base_out_file):
        raise NotImplementedError()

    def _get_mesh_statistics(self, v, f):
        """
        Return the statistics of a mesh as a python dictionary

        TODO: We use the cpp program to get the mesh statistics as of now
        before PyMesh/PyMesh#247  being integrated in upstream
        this is a quick hack as of now.

        """
        mesh_file = os.path.join("/tmp", str(uuid.uuid4()) + ".stl")
        pymesh.save_mesh_raw(mesh_file, v, f)
        cout_mesh_statistics = (
            os.popen(f"mesh_statistics -i {mesh_file}").read().split("\n")[:-1]
        )
        # cout_mesh statistics is a list of string of the form:
        # Name of statistic: value
        # here we parse it to get a dictionary of the item:
        #  {"name_of_statistic": value, …}
        mesh_statistics = {
            t[0].strip().lower().replace(" ", "_"): float(t[1])
            for t in map(lambda x: x.split(":"), cout_mesh_statistics)
        }

        return mesh_statistics

    def run(self, tif_stack_file, out_folder):
        os.makedirs(out_folder, exist_ok=True)

        # path/to/file.name.ext file.name
        basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

        base_out_file = os.path.join(out_folder, basename)

        logging.info(f"→ Image segmentation")
        occupancy_map = self._extract_occupancy_map(tif_stack_file, base_out_file)

        assert (
            0 <= occupancy_map.min()
        ), f"The occupancy map values should be between 0 and 1, currently: {occupancy_map.min()}"
        assert (
            occupancy_map.max() <= 1
        ), f"The occupancy map values should be between 0 and 1, currently: {occupancy_map.max()}"
        assert (
            len(occupancy_map.shape) == 3
        ), f"The occupancy map values should be a 3D array, currently: {len(occupancy_map.shape)}"

        logging.info(f"Occupancy map info")
        logging.info(f"  min        : {occupancy_map.min()}")
        logging.info(f"  max        : {occupancy_map.max()}")
        logging.info(f"  shape      : {occupancy_map.shape}")

        if self.segment_occupancy_map:
            logging.info(f"Segmenting occupancy map info on level: {self.level}")
            occupancy_map = np.array(occupancy_map > self.level, dtype=np.uint8)
            # Remove inner part which are lower that the current level
            occupancy_map = flood_fill(occupancy_map, (1, 1, 1), 2)
            # Create a segmented occupancy map with 2 homogeneous values
            occupancy_map = (self.level + 10e-3) * (occupancy_map != 2)
            logging.info(f"Segmented occupancy map info")
            logging.info(f"  min        : {occupancy_map.min()}")
            logging.info(f"  max        : {occupancy_map.max()}")
            logging.info(f"  shape      : {occupancy_map.shape}")

        if self.save_occupancy_map:
            surface_file = base_out_file + "_occupancy_map.tif"
            occupancy_map_int = np.array(occupancy_map * 255, dtype=np.uint8)
            logging.info(f"Saving extracted occupancy map in: {surface_file}")
            io.imsave(surface_file, occupancy_map_int)

        logging.info(f"→ Isosurface extraction")
        logging.info(f"  Level      : {self.level}")
        logging.info(f"  Spacing    : {self.spacing}")
        logging.info(f"  Step-size  : {self.step_size}")

        v, f, normals, values = measure.marching_cubes(
            occupancy_map,
            level=self.level,
            spacing=(self.spacing, self.spacing, self.spacing),
            gradient_direction=self.gradient_direction,
            step_size=self.step_size,
            # we enforce non-degeneration
            allow_degenerate=True,
            mask=None,
        )

        logging.info(f"Extracted mesh")
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces: {len(f)}")

        if self.save_temp:
            extracted_mesh_file = base_out_file + "_extracted_mesh.stl"
            logging.info(f"Saving extracted mesh in: {extracted_mesh_file}")
            pymesh.save_mesh_raw(extracted_mesh_file, v, f)

        mesh = pymesh.meshio.form_mesh(v, f)

        logging.info(f"→ Splitting mesh in connected components")
        meshes = pymesh.separate_mesh(mesh, connectivity_type="auto")
        logging.info(f"  {len(meshes)} connected components")
        logging.info("")

        for i, m in enumerate(meshes):
            vi = m.vertices
            fi = m.faces

            logging.info(f"Connected component #{i+1}")
            logging.info(f"  Vertices: {len(vi)}")
            logging.info(f"  Faces   : {len(fi)}")
            logging.info("")
            if self.save_temp:
                cc_mesh_file = extracted_mesh_file.replace(".stl", f"_{i}.stl")
                logging.info(f"Saving connected component #{i}: {cc_mesh_file}")
                pymesh.save_mesh_raw(cc_mesh_file, vi, fi)

        # Taking the main mesh
        # Once again, we take the first connected component
        id_main_component = np.argmax([mesh.num_vertices for mesh in meshes])
        mesh_to_simplify = meshes[id_main_component]

        logging.info(f"→ Mesh decimation")
        decimated_mesh = self._mesh_decimation(mesh_to_simplify)
        logging.info(f"Decimated mesh")
        logging.info(f"  Vertices: {len(decimated_mesh.vertices)}")
        logging.info(f"  Faces   : {len(decimated_mesh.faces)}")

        logging.info(f"→ Mesh repairing")
        final_mesh = self._mesh_repairing(decimated_mesh)
        logging.info(f"Final mesh")
        logging.info(f"  Vertices: {len(final_mesh.vertices)}")
        logging.info(f"  Faces   : {len(final_mesh.faces)}")

        final_mesh_file = base_out_file + "_final_mesh.stl"
        pymesh.save_mesh_raw(final_mesh_file, final_mesh.vertices, final_mesh.faces)
        logging.info(f"Saved final mesh in: {final_mesh_file}")

        v = final_mesh.vertices
        f = np.asarray(final_mesh.faces, dtype=np.int32)
        mesh_info = self._get_mesh_quality_info(v, f)

        logging.info("Information of the output mesh:")
        for k, v in mesh_info.items():
            logging.info(f"   {k}: {v}")

        logging.info(f"Pipeline {self.__class__.__name__} done")
        return final_mesh.vertices, final_mesh.faces, mesh_info

    def _get_mesh_quality_info(self, v: np.ndarray, f: np.ndarray):
        """
        Return a mesh statistics and passing tests for a mesh.
        """
        mesh_info = self._get_mesh_statistics(v, f)

        # Test for the mesh usability for shape analysis methods
        try:
            -igl.cotmatrix(v, f)
        except Exception as e:
            print(f" ❌ COT matrix test failed")
            mesh_info["cot_matrix_test"] = "Failed"
            print("Exception:", e)
        else:
            mesh_info["cot_matrix_test"] = "Passed"
        try:
            igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        except Exception as e:
            print(f" ❌ Mass matrix test failed")
            mesh_info["mass_matrix_test"] = "Failed"
            print("Exception:", e)
        else:
            mesh_info["mass_matrix_test"] = "Passed"

        # We only test mesh with a small number of vertices
        n_vertices = v.shape[0]
        if n_vertices < 3000:
            try:
                ind = np.arange(0, n_vertices, dtype=np.int32)
                np.stack(
                    [
                        igl.exact_geodesic(v, f, np.array([i], dtype=np.int32), ind)
                        for i in ind
                    ]
                )
            except Exception as e:
                print(f" ❌ Geodesic matrix test failed")
                mesh_info["geodesic_matrix_test"] = "Failed"
                print("Exception:", e)
            else:
                print(f" ✅ Geodesic matrix test passed")
                mesh_info["geodesic_matrix_test"] = "Passed"
        else:
            mesh_info["geodesic_matrix_test"] = "Not ran"

        mesh_correctness_info = [
            "polygon_polyhedron",
            "triangular_mesh",
            "cot_matrix_test",
            "mass_matrix_test",
            "self_intersecting",
            "non_manifold_vertices",
            "degenerated_faces",
            "mesh_closed",
            "number_of_components",
            "number_of_borders",
            "genus",
        ]

        reordered_mesh_info = dict()
        reordered_mesh_info["mesh_correctness"] = {
            info: mesh_info[info] for info in mesh_correctness_info
        }
        reordered_mesh_info["mesh_statistics"] = {
            k: v for k, v in mesh_info.items() if k not in mesh_correctness_info
        }

        return reordered_mesh_info

    def _mesh_decimation(self, mesh):
        """
        Remesh the input mesh to remove degeneracies and improve triangle quality.

        Taken and adapted from:
        https://github.com/PyMesh/PyMesh/blob/master/scripts/fix_mesh.py

        TODO: to adapt and calibrate
        """
        bbox_min, bbox_max = mesh.bbox
        diag_len = norm(bbox_max - bbox_min)
        if self.detail == "normal":
            target_len = diag_len * 5e-3
        elif self.detail == "high":
            target_len = diag_len * 2.5e-3
        elif self.detail == "low":
            target_len = diag_len * 1e-2
        logging.info(f"Target resolution: {target_len} mm")

        count = 0
        mesh, __ = pymesh.remove_degenerated_triangles(mesh, num_iterations=100)
        mesh, __ = pymesh.split_long_edges(mesh, target_len)
        num_vertices = mesh.num_vertices
        while True:
            mesh, __ = pymesh.collapse_short_edges(mesh, abs_threshold=1e-6)
            mesh, __ = pymesh.collapse_short_edges(
                mesh, abs_threshold=target_len, preserve_feature=True
            )
            mesh, __ = pymesh.remove_obtuse_triangles(
                mesh, max_angle=150.0, max_iterations=100
            )
            if mesh.num_vertices == num_vertices:
                break

            num_vertices = mesh.num_vertices
            logging.info("# Number of vertices: {}".format(num_vertices))
            count += 1
            if count > 5:
                break

        mesh = pymesh.resolve_self_intersection(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh = pymesh.compute_outer_hull(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_obtuse_triangles(
            mesh, max_angle=179.0, max_iterations=5
        )
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        meshes = pymesh.separate_mesh(mesh, connectivity_type="auto")

        # Once again, we take the first connected component
        id_main_component = np.argmax([mesh.num_vertices for mesh in meshes])
        final_mesh = meshes[id_main_component]

        return final_mesh

    @staticmethod
    def _mesh_repairing(mesh):
        """
        Fix the mesh to get to a 2-manifold:
         - no self-intersection
         - closed mesh
        """
        #
        v = np.copy(mesh.vertices)
        f = np.copy(mesh.faces)

        logging.info(f"Fixing mesh")
        meshfix = pymeshfix.MeshFix(v, f)
        meshfix.repair()
        logging.info(f"Fixed mesh")

        mesh = pymesh.meshio.form_mesh(meshfix.v, meshfix.f)

        return mesh


class DirectMeshingPipeline(OPT2MeshPipeline):
    """
    Directly mesh a raw file and perform the simplification
    of it then.
    """

    def _extract_occupancy_map(self, file, base_out_file):

        if ".h5" in file:
            hf = h5py.File(file, "r")
            key = list(hf.keys())[0]
            occupancy_map = np.array(hf[key])
        elif ".tif" in file:
            occupancy_map = io.imread(file)

        if occupancy_map.max() > 1 and occupancy_map.dtype != np.float:
            # uint8 [[0,255]] to float32 [0,1] representation
            occupancy_map = occupancy_map / 255

        return occupancy_map
