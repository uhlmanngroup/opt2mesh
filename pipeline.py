import logging
import os
import time
import uuid
from abc import abstractmethod, ABC

import igl
import pymesh
import numpy as np
from guppy import hpy
from joblib import Parallel, delayed
from scipy.linalg import norm
from skimage import io, measure

import morphsnakes as ms


class TIF2MeshPipeline(ABC):
    """
    General pipeline to convert TIF stacks of images to a STL mesh file.

    It:
      Loads OPT data
      Extracts surface
      Create Mesh
      Clean, smooth and simplify mesh

    The surface extraction is up to each implementation.

    """

    def __init__(self, iterations=50,
                 level=0.999,
                 spacing=1,
                 gradient_direction="descent",
                 step_size=1,
                 timing=True,
                 detail="high",
                 save_temp=False,
                 on_slices=False,
                 n_jobs=-1
                 ):
        self.iterations: int = iterations
        self.level: float = level
        self.spacing: int = spacing
        self.gradient_direction: str = gradient_direction
        self.step_size: int = step_size
        self.timing: bool = timing
        self.detail: str = detail
        self.save_temp: bool = save_temp
        self.on_slices: bool = on_slices
        self.n_jobs: int = n_jobs

    @abstractmethod
    def _extract_occupancy_map(self, tif_stack_file, base_out_file):
        raise NotImplementedError()

    def get_mesh_statistics(self, v, f):
        """
        Return the statistics of a mesh as a python dictionary

        TODO: We use the cpp program to get the mesh statistics as of now
        before PyMesh/PyMesh#247  being integrated in upstream
        this is a quick hack as of now.

        """
        mesh_file = os.path.join("/tmp", str(uuid.uuid4()) + ".stl")
        igl.write_triangle_mesh(mesh_file, v, f)
        cout_mesh_statistics = os.popen(f"mesh_statistics -i {mesh_file}").read().split("\n")[:-1]
        # cout_mesh statistics is a list of string of the form:
        # Name of statistic: value
        # here we parse it to get a dictionary of the item:
        #  {"name_of_statistic": value, …}
        mesh_statistics = {
            t[0].strip().lower().replace(" ", "_"): float(t[1]) for
            t in map(lambda x: x.split(":"), cout_mesh_statistics)
        }

        return mesh_statistics

    def run(self, tif_stack_file, out_folder):
        os.makedirs(out_folder, exist_ok=True)

        # path/to/file.name.ext file.name
        basename = ".".join(tif_stack_file.split(os.sep)[-1].split(".")[:-1])

        base_out_file = os.path.join(out_folder, basename)

        logging.info(f"Input file: {tif_stack_file}")

        logging.info(f"Extracting surface")
        occupancy_map = self._extract_occupancy_map(tif_stack_file, base_out_file)

        surface_file = base_out_file + "_surface.tif"
        logging.info(f"Saving extracted surface in: {surface_file}")

        io.imsave(surface_file, occupancy_map)

        logging.info(f"Extracting mesh from surface")
        v, f, normals, values = measure.marching_cubes(occupancy_map,
                                                       level=self.level,
                                                       spacing=(self.spacing, self.spacing, self.spacing),
                                                       gradient_direction=self.gradient_direction,
                                                       step_size=self.step_size,
                                                       # we enforce non-degeneration
                                                       allow_degenerate=True,
                                                       mask=None)

        if self.save_temp:
            raw_mesh_file = base_out_file + "_raw_mesh.stl"
            logging.info(f"Saving extracted mesh in: {raw_mesh_file}")
            igl.write_triangle_mesh(raw_mesh_file, v, f)

        v, f = self.clean_mesh(v, f)

        clean_mesh_file = base_out_file + "_cleaned_mesh.stl"

        if self.save_temp:
            logging.info(f"Saving clean mesh in: {clean_mesh_file}")
            igl.write_triangle_mesh(clean_mesh_file, v, f)

        mesh = pymesh.meshio.form_mesh(v, f)

        logging.info(f"Splitting mesh in connected components")
        meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')
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

        final_mesh_file = base_out_file + "_final_mesh.stl"
        logging.info(f"Saving final simplified mesh in: {final_mesh_file}")
        pymesh.meshio.save_mesh(final_mesh_file, final_output_mesh)
        logging.info(f"Saved final simplified mesh !")
        stats = self.get_mesh_statistics(mesh_to_simplify.vertices, mesh_to_simplify.faces)
        logging.info("Statistics of final simplified mesh:")
        for k, v in stats:
            logging.info(f"{k}: {v}")

        logging.info("Pipeline done!")

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

        logging.info("Output mesh")
        logging.info(f"  Vertices: {len(v)}")
        logging.info(f"  Faces: {len(f)}")
        logging.info('')

        return v, f

    def _mesh_simplification(self, mesh):
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
            mesh, __ = pymesh.collapse_short_edges(mesh, abs_threshold=target_len,
                                                   preserve_feature=True)
            mesh, __ = pymesh.remove_obtuse_triangles(mesh, max_angle=150.0,
                                                      max_iterations=100)
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
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, max_angle=179.0,
                                                  max_iterations=5)
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        meshes = pymesh.separate_mesh(mesh, connectivity_type='auto')

        # Once again, we take the first connected component
        final_mesh = meshes[0]

        return final_mesh


class GACPipeline(TIF2MeshPipeline):

    def __init__(self, gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=50, level=0.999, spacing=1, save_temp=False,
                 on_slices=False, n_jobs=-1,
                 # GAC specifics
                 smoothing=1, threshold="auto", balloon=1, alpha=1000, sigma=5):
        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp, on_slices=on_slices,
                         n_jobs=n_jobs)

        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.alpha = alpha
        self.sigma = sigma

    def _extract_occupancy_map(self, tif_stack_file, base_out_file):
        logging.info(f"Starting Morphological Geodesic Active Contour on the full image")
        logging.info(f"Loading full data")
        opt_data = io.imread(tif_stack_file) / 255.0
        logging.info(f"Loaded full data")

        gradient_image = ms.inverse_gaussian_gradient(opt_data,
                                                      alpha=self.alpha,
                                                      sigma=self.sigma)

        if self.on_slices:
            occupancy_map = np.zeros(gradient_image.shape)

            init_ls = ms.circle_level_set(gradient_image[0].shape)

            start = time.time()
            logging.info(f"Starting Morphological Geodesic Active Contour on slices")
            for i, slice in enumerate(gradient_image):
                logging.info(f"Running Morphological Geodesic Active Contour on slice {i}")

                occupancy_map[i, :, :] = \
                    ms.morphological_geodesic_active_contour(slice,
                                                             iterations=self.iterations,
                                                             init_level_set=init_ls,
                                                             smoothing=self.smoothing,
                                                             threshold=self.threshold,
                                                             balloon=self.balloon)

            end = time.time()
            logging.info(f"Done Morphological Geodesic Active Contour on slices in {(end - start)}s")
        else:
            # Initialization of the level-set.
            init_ls = ms.circle_level_set(opt_data.shape)

            logging.info(f"Running Morphological Geodesic Active Contour on full")

            start = time.time()

            # MorphGAC
            occupancy_map = ms.morphological_geodesic_active_contour(gradient_image,
                                                                     iterations=self.iterations,
                                                                     init_level_set=init_ls,
                                                                     smoothing=self.smoothing,
                                                                     threshold=self.threshold,
                                                                     balloon=self.balloon)
            end = time.time()
            logging.info(f"Done Morphological Geodesic Active Contour on full in {(end - start)}s")
            del opt_data, init_ls

        return occupancy_map


class ACWEPipeline(TIF2MeshPipeline):

    def __init__(self, gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=150, level=0.999, spacing=1, save_temp=False,
                 on_slices=False, n_jobs=-1,
                 # ACWE specific
                 on_halves=False, smoothing=1, lambda1=3, lambda2=1):

        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp, on_slices=on_slices,
                         n_jobs=n_jobs)

        self.on_halves: bool = on_halves
        self.lambda1: int = lambda1
        self.lambda2: int = lambda2
        self.smoothing: int = smoothing

    def _extract_occupancy_map(self, tif_stack_file, base_out_file):
        if self.on_halves:
            logging.info(f"Starting Morphological Chan Vese on halves")
            self._tif2morphsnakes_halves(tif_stack_file, base_out_file)
            logging.info(f"Done Morphological Chan Vese on halves")
            occupancy_map = self._morphsnakes_halves2surface(base_out_file)
        elif self.on_slices:
            logging.info(f"Starting Morphological Chan Vese on slices")
            occupancy_map = self._tif2morphsnakes_slices(tif_stack_file)
            logging.info(f"Done Morphological Chan Vese on slices")
        else:
            logging.info(f"Starting Morphological Chan Vese on the full image")
            logging.info(f"Loading full data")
            opt_data = io.imread(tif_stack_file)
            logging.info(f"Loaded full data")

            # Initialization of the level-set.
            init_ls = ms.circle_level_set(opt_data.shape)

            logging.info(f"Running Morphological Chan Vese on full")

            start = time.time()

            occupancy_map = ms.morphological_chan_vese(opt_data,
                                                       init_level_set=init_ls,
                                                       iterations=self.iterations,
                                                       smoothing=self.smoothing,
                                                       lambda1=self.lambda1,
                                                       lambda2=self.lambda2)
            end = time.time()
            logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")
            del opt_data, init_ls

        return occupancy_map

    def __acwe_on_one_half(self, tif_stack_file, base_out_file, suffix):

        # TODO: the half_size index should adapt to the data shape
        # the cube has a size of (511,512,512)
        half_size_index = 256

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
        logging.info(f"Done Morphological Chan Vese on {suffix} in {(end - start)}s")

        half_surface_file = base_out_file + f"_{suffix}.tif"

        logging.info(f"Saving half {suffix} in: {half_surface_file}")
        io.imsave(half_surface_file, half_surface)

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

        Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(self.__acwe_on_one_half(
                tif_stack_file, base_out_file, suffix)) for suffix in
            ["x_front", "x_back", "y_front", "y_back", "z_front", "z_back"]
        )

    def _tif2morphsnakes_slices(self, tif_stack_file):
        """
        Create morphsnakes surfaces on each slices of the cubes independently.

        :param tif_stack_file: path to the TIF stack to process
        """

        h = hpy()
        logging.info("Before loading the data")
        logging.info(str(h.heap()))

        opt_data = io.imread(tif_stack_file)
        occupancy_map = np.zeros(opt_data.shape)

        init_ls = ms.circle_level_set(opt_data[0].shape)

        start = time.time()
        logging.info(f"Starting Morphological Chan Vese on slices")
        for i, slice in enumerate(opt_data):
            logging.info(f"Running Morphological Chan Vese on slice {i}")

            occupancy_map[i, :, :] = ms.morphological_chan_vese(slice,
                                                                init_level_set=init_ls,
                                                                iterations=self.iterations,
                                                                smoothing=self.smoothing,
                                                                lambda1=self.lambda1,
                                                                lambda2=self.lambda2)

        end = time.time()
        logging.info(f"Done Morphological Chan Vese on slices in {(end - start)}s")

        return occupancy_map

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
        occupancy_map = (x_front_reshaped + x_back_reshaped + y_back_reshaped
                         + y_front_reshaped + z_back_reshaped + z_front_reshaped).clip(0, 1)

        if self.save_temp:
            io.imsave(base_out_file + "_x_front_reshaped.tif", x_front_reshaped)
            io.imsave(base_out_file + "_x_back_reshaped.tif", x_back_reshaped)

            io.imsave(base_out_file + "_y_front_reshaped.tif", y_front_reshaped)
            io.imsave(base_out_file + "_y_back_reshaped.tif", y_back_reshaped)

            io.imsave(base_out_file + "_z_front_reshaped.tif", z_front_reshaped)
            io.imsave(base_out_file + "_z_back_reshaped.tif", z_back_reshaped)

        return occupancy_map


class AutoContextPipeline(TIF2MeshPipeline):

    def __init__(self, project,
                 gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=150, level=0.999, spacing=1,
                 save_temp=False, on_slices=False, n_jobs=-1,
                 # Autocontext specific
                 on_halves=False):

        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp, on_slices=on_slices,
                         n_jobs=n_jobs)

        self.on_halves: bool = on_halves
        self.project: str = project
        self.output_filename_format: str = "/tmp/{nickname}/{nickname}{slice_index}_pred.tif "

        self._drange = '"(0,255)"'
        self._dtype = "uint8"
        self._output_format = 'tif sequence'

    @property
    def _ilastik_out_folder(self):
        # /path/to/folder/file.ext → /path/to/folder/
        return os.sep.join(self.output_filename_format.split(os.sep)[:-1])

    def _extract_occupancy_map(self, tif_stack_file, base_out_file):

        # Need some config to have it accessible here
        command = "ilastik "
        command += "--headless "
        command += f"--project={self.project} "
        command += f"--output_format={self._output_format} "
        command += f"--output_filename_format={self.output_filename_format} "
        command += f"--export_dtype={self._dtype} "
        command += f'--export_drange={self._drange} '
        command += f'--pipeline_result_drange={self._drange} '

        # /full/path/to/MNS_M897_115_clahe_nl_means_denoised_*.tif"
        command += tif_stack_file

        out_ilastik = os.popen(command).read()

        logging.info("Ilastik cout:")
        logging.info(out_ilastik)

        # Slices have been dropped on disc here, we are performing some
        # reconstruction here to get access to the segmentation then
        ilastik_out_folder = os.sep.join(self.output_filename_format.split(os.sep)[:-1])
        files = sorted(os.listdir(ilastik_out_folder))
        occupancy_map = np.array([io.imread(f) for f in files], dtype=np.uint8)

        return occupancy_map
