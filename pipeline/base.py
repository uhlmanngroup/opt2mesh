import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from itertools import chain
from typing import List

import h5py
import igl
import numpy as np
import pymeshfix
from skimage import io, measure, exposure, filters
from skimage.morphology import flood_fill


class OPT2MeshPipeline(ABC):
    """
    General pipeline to convert a OPT to a mesh.

    Input: TIF Stack
    Output: STL file

    This pipeline:
      Loads the OPT scan
      Preprocesses the OPT scan (not by default)
      Segments the OPT scan
      Extracts the mesh using Lewiner's variant of Marching Cubes
      Corrects and simplifies the mesh

    """

    def __init__(
        self,
        level=0.5,
        spacing=1,
        gradient_direction="descent",
        step_size=1,
        detail=6000,
        save_temp=False,
        segment_occupancy_map=True,
        save_occupancy_map=False,
        align_mesh=False,
        preprocess_opt_scan=False,
        loops_to_remove=None,
    ):
        """
        @param level: the iso-value to extract from the volume using marching cube
            See skimage.measure.marching_cubes docstring for more information.
        @param spacing: the spacing in the volume
            See skimage.measure.marching_cubes docstring for more information.
        @param gradient_direction: string
            Controls if the mesh was generated from an isosurface with gradient
            descent toward objects of interest (the default), or the opposite,
            considering the *left-hand* rule.
            The two options are:
            * descent : Object was greater than exterior
            * ascent : Exterior was greater than object


        """
        self.level: float = level
        self.spacing: int = spacing
        self.gradient_direction: str = gradient_direction
        self.step_size: int = step_size
        self.detail = detail
        self.save_temp: bool = save_temp
        self.segment_occupancy_map: bool = segment_occupancy_map
        self.save_occupancy_map: bool = save_occupancy_map
        self.align_mesh: bool = align_mesh
        self.preprocess_opt_scan: bool = preprocess_opt_scan
        self.loops_to_remove: str = loops_to_remove

    @abstractmethod
    def _extract_occupancy_map(
        self, opt2process: np.ndarray, base_out_file: str
    ) -> np.ndarray:
        raise NotImplementedError()

    def _preprocessing(self, opt_scan: np.ndarray) -> np.ndarray:
        """
        Perform a preprocessing with contrast adjustment and denoising.

        This step is made optional in the pipeline.

        Constrast Limited Adaptive Histogram Equalization is used for
        the constrast correction.

        Median Filtering is used for the denoising.

        """
        logging.info(f"Performing full preprocessing")

        opt_data_adapt_eq = exposure.equalize_adapthist(
            opt_scan, clip_limit=0.03
        )
        opt_data_adapt_eq = (opt_data_adapt_eq * 255).astype(np.uint8)

        denoised_opt_data = filters.median(opt_data_adapt_eq)

        logging.info(f"Cropping volume")
        return denoised_opt_data

    def _get_mesh_statistics(self, v: np.ndarray, f: np.ndarray) -> dict:
        """
        Return the statistics of a mesh as a python dictionary

        TODO: We use the cpp program to get the mesh statistics as of now
        before PyMesh/PyMesh#247  being integrated in upstream
        this is a quick hack as of now.

        This uses the mesh_statistics executable from this C++ code here:
        https://gitlab.ebi.ac.uk/jerphanion/mesh-processing-pipeline/-/tree/master/src/cpp/pipeline

        @param v: array of vertices
        @param f: array of faces
        """
        with tempfile.TemporaryDirectory() as tmp:
            mesh_file = os.path.join(tmp, "mesh.stl")
            igl.write_triangle_mesh(mesh_file, v, f)
            # NOTE (executable)
            cout_mesh_statistics = (
                os.popen(f"mesh_statistics -i {mesh_file}")
                .read()
                .split("\n")[:-1]
            )
        # cout_mesh statistics is a list of string of the form:
        # Name of statistic: value
        # here we parse it to get a dictionary of the item:
        #  {"name_of_statistic": value, ???}
        mesh_statistics = {
            t[0].strip().lower().replace(" ", "_"): float(t[1])
            for t in map(lambda x: x.split(":"), cout_mesh_statistics)
        }

        return mesh_statistics

    def _load_opt_scan(self, opt_file: str) -> np.ndarray:
        if ".h5" in opt_file:
            hf = h5py.File(opt_file, "r")
            key = list(hf.keys())[0]
            raw_opt = np.array(hf[key])
        elif ".tif" in opt_file:
            raw_opt = io.imread(opt_file)
        return raw_opt

    def _separate_mesh(self, v: np.ndarray, f: np.ndarray) -> list:
        """
        Separate the mesh in different connected components.

        @param v: array of vertices
        @param f: array of faces

        Returns: a list of tuples of array of vertices and array of faces.
        """
        v, a, b, f = igl.remove_duplicate_vertices(v, f, epsilon=10e-7)

        v_indices = igl.vertex_components(f)
        f_indices = igl.face_components(f)

        cc_indices = np.unique(v_indices)

        meshes = list()

        for cc_index in cc_indices:
            v_cc = v[v_indices == cc_index, :]
            f_cc = f[f_indices == cc_index, :]

            (i_v_cc,) = np.where(v_indices == cc_index)
            vertices_remapping = dict(zip(i_v_cc, range(len(i_v_cc))))

            mp = np.vectorize(
                lambda entry: vertices_remapping.get(entry, entry)
            )

            f_cc = mp(f_cc)

            meshes.append((v_cc, f_cc))

        return meshes

    def run(
        self, opt_file: str, out_folder: str
    ) -> (np.ndarray, np.ndarray, dict):
        """
        Run the pipeline and extract the mesh.

        Returns the vertices and faces array as well as a dictionary
        of the mesh information.

        @param opt_file: the path to the OPT scan (as a tif file)
        @param out_folder: the output folder to use to store results.
        """
        os.makedirs(out_folder, exist_ok=True)

        raw_opt = self._load_opt_scan(opt_file)

        if self.preprocess_opt_scan:
            logging.info(f"Preprocessing OPT scan")
            opt2process = self._preprocessing(raw_opt)
        else:
            opt2process = raw_opt

        # path/to/file.name.ext file.name
        basename = ".".join(opt_file.split(os.sep)[-1].split(".")[:-1])
        base_out_file = os.path.join(out_folder, basename)

        logging.info(f"??? Image segmentation")
        occupancy_map = self._extract_occupancy_map(opt2process, base_out_file)

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
            if self.save_occupancy_map:
                surface_file = (
                    base_out_file + "_occupancy_map_before_segmentation.tif"
                )
                occupancy_map_int = np.array(
                    occupancy_map * 255, dtype=np.uint8
                )
                logging.info(
                    "Saving extracted occupancy before segmentation"
                    f"in: {surface_file}"
                )
                io.imsave(surface_file, occupancy_map_int)

            logging.info(
                f"Segmenting occupancy map info on level: {self.level}"
            )
            occupancy_map = np.array(
                occupancy_map > self.level, dtype=np.uint8
            )
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

        logging.info(f"??? Isosurface extraction")
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

        if self.align_mesh:
            # For some reasons, the mesh which is extracted using
            # the marching cubes implementation is not matching the original
            # orientation.
            logging.info(
                "Aligning the mesh on the original OPT scan orientation"
            )
            from scipy.spatial.transform import Rotation

            rotation_mat = Rotation.from_euler("y", 90, degrees=True)
            v = v @ rotation_mat.as_matrix().T

        if self.save_temp:
            extracted_mesh_file = base_out_file + "_extracted_mesh.stl"
            logging.info(f"Saving extracted mesh in: {extracted_mesh_file}")
            igl.write_triangle_mesh(extracted_mesh_file, v, f)

        logging.info(f"??? Splitting mesh in connected components")
        meshes = self._separate_mesh(v, f)
        logging.info(f"  {len(meshes)} connected components")
        logging.info("")

        for i, (vi, fi) in enumerate(meshes):
            logging.info(f"Connected component #{i+1}")
            logging.info(f"  Vertices: {len(vi)}")
            logging.info(f"  Faces   : {len(fi)}")
            logging.info("")
            if self.save_temp:
                cc_mesh_file = extracted_mesh_file.replace(".stl", f"_{i}.stl")
                logging.info(
                    f"Saving connected component #{i}: {cc_mesh_file}"
                )
                igl.write_triangle_mesh(cc_mesh_file, vi, fi)

        # Taking the main mesh
        # Once again, we take the first connected component
        id_main_component = np.argmax([vi.shape[0] for (vi, fi) in meshes])
        v, f = meshes[id_main_component]

        logging.info(f"??? Mesh decimation")
        v, f, succeeded = self._mesh_decimation(v, f)
        logging.info(
            "Decimated mesh" if succeeded else "Failed decimating mesh"
        )
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces   : {len(f)}")

        logging.info(f"??? Mesh repairing")
        v, f = self._mesh_repairing(v, f)

        if self.loops_to_remove is not None:
            logging.info(f" ??? Topological correction of the mesh")
            v, f = self.topological_correction(
                v, f, method=self.loops_to_remove
            )

        f = np.asarray(f, dtype=np.int32)

        logging.info(f"Final mesh")
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces   : {len(f)}")

        final_mesh_file = base_out_file + "_final_mesh.stl"
        igl.write_triangle_mesh(final_mesh_file, v, f)
        logging.info(f"Saved final mesh in: {final_mesh_file}")

        mesh_info = self._get_mesh_quality_info(v, f)

        logging.info(f"Pipeline {self.__class__.__name__} done")
        return v, f, mesh_info

    def _get_mesh_quality_info(self, v: np.ndarray, f: np.ndarray):
        """
        Return the mesh statistics and passing tests for a mesh.
        """
        mesh_info = self._get_mesh_statistics(v, f)

        # Test for the mesh usability for shape analysis methods
        try:
            -igl.cotmatrix(v, f)
        except Exception as e:
            logging.info(f" ??? COT matrix test failed")
            mesh_info["cot_matrix_test"] = "Failed"
            logging.info("Exception:", e)
        else:
            logging.info(f" ??? COT matrix test passed")
            mesh_info["cot_matrix_test"] = "Passed"
        try:
            igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        except Exception as e:
            logging.info(f" ??? Mass matrix test failed")
            mesh_info["mass_matrix_test"] = "Failed"
            logging.info("Exception:", e)
        else:
            logging.info(f" ??? Mass matrix test passed")
            mesh_info["mass_matrix_test"] = "Passed"

        # We only test mesh with a small number of vertices
        n_vertices = v.shape[0]
        if n_vertices < 3000:
            try:
                ind = np.arange(0, n_vertices, dtype=np.int32)
                np.stack(
                    [
                        igl.exact_geodesic(
                            v, f, np.array([i], dtype=np.int32), ind
                        )
                        for i in ind
                    ]
                )
            except Exception as e:
                logging.info(f" ??? Geodesic matrix test failed")
                mesh_info["geodesic_matrix_test"] = "Failed"
                logging.info("Exception:", e)
            else:
                logging.info(f" ??? Geodesic matrix test passed")
                mesh_info["geodesic_matrix_test"] = "Passed"
        else:
            mesh_info["geodesic_matrix_test"] = "Not ran"

        mesh_correctness_info = [
            "polygon_polyhedron",
            "triangular_mesh",
            "cot_matrix_test",
            "mass_matrix_test",
            "geodesic_matrix_test",
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
            info: mesh_info.get(info, "Not computed")
            for info in mesh_correctness_info
        }
        reordered_mesh_info["mesh_statistics"] = {
            k: v
            for k, v in mesh_info.items()
            if k not in mesh_correctness_info
        }

        return reordered_mesh_info

    def _mesh_decimation(
        self, v: np.ndarray, f: np.ndarray
    ) -> (np.array, np.array, bool):
        """
        Decimate the mesh by edges collapsing.
        """

        if self.detail == "original":
            return v, f, True

        if isinstance(self.detail, int):
            target_faces_number = self.detail
        elif isinstance(self.detail, str):
            # This is a simple adaptation for retro compatibility
            # but this can be adapted.
            # The original mesh can really be of high resolution
            target_faces_number = {
                "high": int(0.5 * f.shape[0]),
                "normal": int(0.1 * f.shape[0]),
                "low": 3000,
            }[self.detail]
        else:
            raise RuntimeError(
                "detail should be an integer or a string in ['low', 'normal', 'high']"
            )

        logging.info(f"Decimating the mesh to get {target_faces_number} faces")
        if f.shape[0] < target_faces_number:
            logging.info(
                f"No need for decimation; current number of faces: {f.shape[0]}"
            )
            return v, f, True

        succeeded, vp, fp, i_fp, i_vp = igl.decimate(v, f, target_faces_number)

        if succeeded:
            return vp, fp, succeeded
        else:
            # vp and fp are empty in this case sometimes
            # hence we return the original arrays directly
            return v, f, succeeded

    @staticmethod
    def _mesh_repairing(
        v: np.ndarray, f: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Fix the mesh to get to a 2-manifold:
         - no self-intersection
         - closed mesh
        """

        logging.info(f"Fixing mesh")
        meshfix = pymeshfix.MeshFix(v, f)
        meshfix.repair()
        logging.info(f"Fixed mesh")

        return meshfix.v, meshfix.f

    @staticmethod
    def __remove_loops(
        vertices_lines: List[str],
        faces_lines: List[str],
        loops_vertices_indices: List[int],
    ) -> (List[str], List[str]):
        """
        Remove vertices for a mesh information.

        @param vertices_lines: list of string of the form "x y z"
        @param faces_lines: list of string of the form "3 id_a, id_b, id_c" where id_i is the id of the ith vertex
        @param loops_vertices_indices: list of indices of vertices associated to loops
        """
        # remove associated faces/edges:
        vertex_to_remove_str = list(map(str, loops_vertices_indices))
        vertices_to_keep_info = [
            vertex_info
            for i, vertex_info in enumerate(vertices_lines)
            if i not in loops_vertices_indices
        ]

        def face_is_to_keep(line):
            for v in vertex_to_remove_str:
                if v in line:
                    return False
            return True

        a = 0
        n_vertices = len(vertices_lines)
        dec_table = [0] * n_vertices
        for v in range(n_vertices):
            if v in loops_vertices_indices:
                a += 1
            dec_table[v] = v - a

        new_vertices_mapping = dict(zip(list(range(n_vertices)), dec_table))

        def remap_face_vertices(face_line: str):
            """
            We remap the face vertices.

            @param face_line: string of the form "3 id_a id_b id_c"
            """
            _, id_a, id_b, id_c = (
                int(i) for i in face_line.replace("\n", "").split(" ")
            )

            ra, rb, rc = (
                new_vertices_mapping[id_a],
                new_vertices_mapping[id_b],
                new_vertices_mapping[id_c],
            )

            return f"3 {ra} {rb} {rc}\n"

        faces_to_keep_info = list(filter(face_is_to_keep, faces_lines))

        remapped_faces = list(map(remap_face_vertices, faces_to_keep_info))

        return vertices_to_keep_info, remapped_faces

    @staticmethod
    def topological_correction(
        v: np.array, f: np.array, method="handle"
    ) -> (np.array, np.array):
        """
        Remove vertices and faces of loops associated to extra topological features
        on the mesh.

        Uses the ReebHanTun method to identify loops of handles and tunnels.
        Then remove the associated vertices and faces of the handles (default)
        or tunnels (if specified by use_tunnels).

        @param: v: the array of vertices
        @param: f: the array of faces
        @param: method: method to use to remove loops.
        """

        # Using a temp directory for the input/output of the ReebHanTun executable

        # NOTE (executable): ReebHanTun executable
        reebhantun_executable = "ReebHanTun"
        if not shutil.which(reebhantun_executable):
            logging.warning(
                "The executable provided for the Reeb Han Tun method "
                f"({reebhantun_executable}) is not present "
                f"in the path. The mesh is returned as is."
            )
            return v, f

        with tempfile.TemporaryDirectory() as tmpdir:
            basename = "tempmesh"

            # Convert to off because ReebHanTun needs such files as inputs
            off_filename = os.path.join(tmpdir, f"{basename}.off")
            igl.write_triangle_mesh(off_filename, v, f)
            rht_input_mesh = off_filename

            command = (
                f"{reebhantun_executable} -I {rht_input_mesh} -O {basename}"
            )

            # This script outputs those two files here
            loops_lop = f"loops_{basename}.lop"
            loops_list = f"loops_{basename}.list"

            while not (os.path.isfile(loops_lop)):
                cout_rht = os.popen(command).read()
                logging.info("ReebHanTun output:")
                logging.info(cout_rht)

                if " ---- MESH HAS GENUS 0!" in cout_rht:
                    # No need to correct, we just return the mesh
                    return v, f

            # We just move them under the same temp directory to have it all sorted
            os.popen(f"mv {loops_lop} {tmpdir}").read()
            os.popen(f"mv {loops_list} {tmpdir}").read()

            # This files stores the vertices indices associated to loops
            with open(os.path.join(tmpdir, loops_lop), "r") as fp:
                loops_lines = fp.readlines()

            def extract_vertices_index(loop_line: str) -> List[int]:
                """Conversion of string to list

                For instance:
                    loop_line: "handle loop 0size(4): 501,502,535,536,\n"
                    vertices_indices: [501, 502, 535, 536]
                """
                vertices_indices = [
                    int(s)
                    for s in loop_line.replace("\n", "")
                    .split(":")[-1]
                    .split(",")[:-1]
                ]
                return vertices_indices

            def is_handle_loop(loop_line: str) -> bool:
                return loop_line.startswith("handle loop")

            def is_tunnel_loop(loop_line: str) -> bool:
                return loop_line.startswith("tunnel loop")

            # We gather the loops there
            handles_loops: List[List[int]] = list(
                map(
                    extract_vertices_index, filter(is_handle_loop, loops_lines)
                )
            )
            tunnels_loops: List[List[int]] = list(
                map(
                    extract_vertices_index, filter(is_tunnel_loop, loops_lines)
                )
            )

            # We flatten them
            handles_vertices: List[int] = sorted(
                set(chain.from_iterable(handles_loops))
            )
            tunnels_vertices: List[int] = sorted(
                set(chain.from_iterable(tunnels_loops))
            )

            if method == "tunnels":
                vertices_to_remove = tunnels_vertices
            elif method == "handles":
                vertices_to_remove = handles_vertices
            else:
                raise RuntimeError(f"Wrong method: {method}")

            # Extract the information from the OFF format
            with open(rht_input_mesh, "r") as fp:
                off_lines = fp.readlines()
                # Remove potential comment in the file
                off_lines = list(
                    filter(lambda x: not (x.startswith("#")), off_lines)
                )

            mesh_info_line = off_lines[1].split(" ")
            n_vertices = int(mesh_info_line[0])
            vertices_lines = off_lines[2 : n_vertices + 2]
            faces_lines = off_lines[n_vertices + 2 :]

            (
                new_vertices_lines,
                new_faces_lines,
            ) = OPT2MeshPipeline.__remove_loops(
                vertices_lines,
                faces_lines,
                loops_vertices_indices=vertices_to_remove,
            )

            # We just dump the mesh under the off format here before loading it again
            new_off_lines = (
                [
                    "OFF\n",
                    f"{len(new_vertices_lines)} {len(new_faces_lines)} 0\n",
                ]
                + new_vertices_lines
                + new_faces_lines
            )

            fm = os.path.join(tmpdir, basename + "_t.off")

            with open(fm, "w") as fp:
                fp.writelines(new_off_lines)

            v, f = igl.read_triangle_mesh(filename=fm)

            # We finally fair the mesh
            meshfix = pymeshfix.MeshFix(v, f)
            meshfix.repair()

            return meshfix.v, meshfix.f


class DirectMeshingPipeline(OPT2MeshPipeline):
    """
    Directly mesh a raw file and perform the simplification
    of it then.

    See OPT2MeshPipeline.__init__ documentation for the documentation
    of the parameters constructor.
    """

    def _extract_occupancy_map(self, opt2process, base_out_file):
        occupancy_map = opt2process
        if occupancy_map.max() > 1 and occupancy_map.dtype != np.float:
            # uint8 [[0,255]] to float32 [0,1] representation
            occupancy_map = occupancy_map / 255

        return occupancy_map
