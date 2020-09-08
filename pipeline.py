import logging
import os
import time
import uuid
from abc import abstractmethod, ABC

import h5py
import igl
import numpy as np
import pymesh
import torch
import torch.nn.functional as F
from PIL import Image
from guppy import hpy
from joblib import Parallel, delayed
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from skimage import io, measure
from skimage.morphology import dilation, erosion
from torchvision import transforms

import morphsnakes as ms
from dataset import BasicDataset
from scripts.preprocessing import to_hdf5, _fill_binary_image
from unet import UNet


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

        logging.info(f"Extracting occupancy map")
        occupancy_map = self._extract_occupancy_map(tif_stack_file, base_out_file)

        if self.save_temp:
            surface_file = base_out_file + "_occupancy_map.tif"
            occupancy_map_int = np.array(occupancy_map * 255, dtype=np.uint8)
            logging.info(f"Saving extracted occupancy map in: {surface_file}")
            io.imsave(surface_file, occupancy_map_int)

        assert 0 <= occupancy_map.min(), "The occupancy map values should be between 0 and 1"
        assert occupancy_map.max() <= 1, "The occupancy map values should be between 0 and 1"

        logging.info(f"Extracting mesh from occupancy map")
        logging.info(f"   Level      : {self.level}")
        logging.info(f"   Spacing    : {self.spacing}")
        logging.info(f"   Step-size  : {self.step_size}")
        logging.info(f"Extracting mesh from occupancy map")
        logging.info(f"Occupancy map values")
        logging.info(f"    min        : {occupancy_map.min()}")
        logging.info(f"    max        : {occupancy_map.max()}")
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
        # Once again, we take the first connected component
        id_main_component = np.argmax([mesh.num_vertices for mesh in meshes])
        mesh_to_simplify = meshes[id_main_component]

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
        id_main_component = np.argmax([mesh.num_vertices for mesh in meshes])
        final_mesh = meshes[id_main_component]

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

            init_ls = ms.ellipsoid_level_set(gradient_image[0].shape)

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
            init_ls = ms.ellipsoid_level_set(opt_data.shape)

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
            init_ls = ms.ellipsoid_level_set(opt_data.shape)

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
        init_ls = ms.ellipsoid_level_set(opt_data.shape)

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

        init_ls = ms.ellipsoid_level_set(opt_data[0].shape)

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
    """
    Use ilastik for the segmentation using the headless mode.

    All the options and some current problems are specified here:

    https://www.ilastik.org/documentation/basics/headless
    """

    def __init__(self,
                 # Autocontext specific
                 project,
                 use_probabilities=True,
                 #
                 gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=150, level=0.999, spacing=1,
                 save_temp=False, on_slices=False, n_jobs=-1):
        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp,
                         on_slices=on_slices, n_jobs=n_jobs)

        self.project: str = project

        self._use_probabilities = use_probabilities

        if self._use_probabilities:
            # TODO: tune this
            self.level = 0.90

    def _dump_hdf5_on_disk(self, tif_file, base_out_file):
        """
        Convert a tif file to a hdf5 file.

        @param tif_file: path to the tif file
        @param base_out_file: base name for the output file
        @return: the path to the file created
        """
        logging.info(f"Converting {tif_file} to hdf5")
        opt_data = io.imread(tif_file)
        basename = tif_file.split(os.sep)[-1].split(".")[0]
        file_basename = f"{base_out_file}/autocontext/{basename}"
        os.makedirs(f"{base_out_file}/autocontext/", exist_ok=True)
        h5_file = to_hdf5(opt_data, file_basename=file_basename)
        logging.info(f"Dumped hdf5 dataset to {h5_file}")

        return h5_file

    def _post_processing(self, interior_segmentation):
        """
        The result might be noisy after the prediction with Autocontext.
        Hence, we perform some morphological operations on the segmentation
        data.

        @param interior_segmentation: 3D np.array of the segmentation

        @return:
        """
        improved_seg_data = dilation(erosion(dilation(
            gaussian_filter(interior_segmentation, sigma=0.1)))) \
            .astype(np.uint8)
        for i in range(improved_seg_data.shape[0]):
            improved_seg_data[i, :, :] = _fill_binary_image(improved_seg_data[i, :, :])

        return improved_seg_data

    def _extract_occupancy_map(self, tif_file, base_out_file):

        ilastik_output_folder = f"{base_out_file}/autocontext/predictions/"

        # We use h5 here because it is more memory efficient
        # https://forum.image.sc/t/notable-memory-usage-difference-when-running-ilastik-in-headless-mode-on-different-machines/41144/4
        output_format = 'hdf5'
        drange = '"(0,255)"'
        dtype = "uint8"
        output_filename_format = ilastik_output_folder + "{nickname}_pred.h5 "
        in_files = self._dump_hdf5_on_disk(tif_file, base_out_file)

        # Note: one may need some config to have ilastik accessible in PATH
        command = "ilastik "
        command += "--headless "
        command += f"--project={self.project} "
        command += f"--output_format={output_format} "
        command += f"--output_filename_format={output_filename_format} "
        if self._use_probabilities:
            # TODO: change the number when using AutoContext with more stages here
            command += '--export_source="Probabilities Stage 2" '
        else:
            command += f"--export_dtype={dtype} "
            command += f'--export_drange={drange} '
            command += f'--pipeline_result_drange={dtype} '
        command += in_files

        # To have a dedicated file for Ilastik's standard output
        command += f" | tee {ilastik_output_folder + 'ilastik_cli_call.log'}"

        logging.info("Lauching Ilastik")
        logging.info("CLI command:")
        logging.info(command)

        # Running the segmentation on ilastik
        out_ilastik = os.popen(command).read()
        logging.info("Ilastik CLI Call standard output:")
        logging.info(out_ilastik)

        segmentation_file = os.path.join(ilastik_output_folder, os.listdir(ilastik_output_folder)[0])
        assert segmentation_file.endswith(".h5"), f"Not a correct hdf5 file : {segmentation_file}"

        hf = h5py.File(segmentation_file, 'r')
        # We are only interested in the "interior" information.
        # It is the last label, hence the use of "-1"
        noisy_occupancy_map = np.array(hf["exported_data"])[..., -1]
        hf.close()

        if not self._use_probabilities:
            # we work on segmentations which we need to clean
            occupancy_map = self._post_processing(noisy_occupancy_map)
        else:
            occupancy_map = noisy_occupancy_map

        if self.save_temp:
            filename = segmentation_file.replace(".h5", f"_occupancy_map.tif")
            logging.info(f"Saving occupancy map in {filename}")
            hf = h5py.File(filename, 'w')
            hf.create_dataset("exported_data", data=occupancy_map, chunks=True)
            hf.close()

        logging.info(f"Done extracting the occupancy map with Autocontext")

        return occupancy_map


class AutoContextACWEPipeline(TIF2MeshPipeline):
    """
    Use AutoContext to extract the occupancy map (probabilities)
    then runs ACWE on the occupancy map to extract the surface.
    """

    def __init__(self,
                 # AutoContextSpecific
                 project,
                 # ACWE specifics
                 smoothing,
                 lambda1,
                 lambda2,
                 ###
                 gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=150, level=0.999, spacing=1,
                 save_temp=False, on_slices=False, n_jobs=-1):
        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp,
                         on_slices=on_slices, n_jobs=n_jobs)

        self.autocontext_pipeline = AutoContextPipeline(project=project,
                                                        use_probabilities=True,
                                                        gradient_direction="descent",
                                                        step_size=step_size,
                                                        timing=timing,
                                                        detail=detail,
                                                        save_temp=save_temp,
                                                        on_slices=on_slices,
                                                        n_jobs=n_jobs)
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _extract_occupancy_map(self, tif_file, base_out_file):
        """
        See AutoContextACWEPipeline docstring.

        @param tif_file:
        @param base_out_file:
        @return:
        """
        logging.info(f"Running Morphological Chan Vese on full")
        start = time.time()
        occupancy_map = self.autocontext_pipeline._extract_occupancy_map(tif_file, base_out_file)
        end = time.time()
        logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")

        # Initialization of the level-set.
        init_ls = ms.ellipsoid_level_set(occupancy_map.shape)

        logging.info(f"Running Morphological Chan Vese on full")

        start = time.time()
        occupancy_map = ms.morphological_chan_vese(occupancy_map,
                                                   init_level_set=init_ls,
                                                   iterations=self.iterations,
                                                   smoothing=self.smoothing,
                                                   lambda1=self.lambda1,
                                                   lambda2=self.lambda2)
        end = time.time()
        logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")

        return occupancy_map


class UNetPipeline(TIF2MeshPipeline):
    """
    Use a 2D UNet to get occupancy map slices on the 3 different axes.
    Predictions are stacked together to get occupancy maps and are then
    averaged to get a better estimated occupancy map.

    Code adapted from:
     - https://github.com/milesial/Pytorch-UNet

    """

    def __init__(self,
                 # UNet specifics
                 model_file,
                 scale_factor=0.5,
                 ###
                 level=0.5,
                 ###
                 gradient_direction="descent", step_size=1, timing=True,
                 detail="high", iterations=150, spacing=1,
                 save_temp=False, on_slices=False, n_jobs=-1):
        super().__init__(iterations=iterations, level=level, spacing=spacing,
                         gradient_direction=gradient_direction, step_size=step_size,
                         timing=timing, detail=detail, save_temp=save_temp,
                         on_slices=on_slices, n_jobs=n_jobs)

        self.model_file = model_file
        self.scale_factor = scale_factor

        # TODO: this is enforced
        self.level = 0.5

    def _predict(self, net, full_img, device):
        net.eval()

        img = torch.from_numpy(BasicDataset.preprocess(full_img, self.scale_factor))

        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)

            if net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img.size[1]),
                    transforms.ToTensor()
                ]
            )

            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask

    def _extract_occupancy_map(self, tif_file, base_out_file):
        logging.info(f"Running 2D UNet on the 3 axis")
        start = time.time()
        img = io.imread(tif_file)

        # TODO: adapt cropping on the long run
        first, last = 0, 511
        img = img[first:last, first:last, first:last]

        pred_x = np.zeros(img.shape)
        pred_y = np.zeros(img.shape)
        pred_z = np.zeros(img.shape)

        h, w, d = img.shape

        net = UNet(n_channels=1, n_classes=1)

        logging.info("Loading model {}".format(self.model_file))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(torch.load(self.model_file, map_location=device))

        logging.info(f'Prediction w.r.t axis x')
        for x in range(h):
            logging.info(f'Slice x: {x}/{h}')
            pred_x[x, :, :] = self._predict(net=net,
                                            full_img=Image.fromarray(img[x, :, :]),
                                            device=device)

        logging.info(f'Prediction w.r.t axis y')
        for y in range(w):
            logging.info(f'Slice y: {y}/{w}')
            pred_y[:, y, :] = self._predict(net=net,
                                            full_img=Image.fromarray(img[:, y, :]),
                                            device=device)

        logging.info(f'Prediction w.r.t axis z')
        for z in range(d):
            logging.info(f'Slice z: {z}/{d}')
            pred_z[:, :, z] = self._predict(net=net,
                                            full_img=Image.fromarray(img[:, :, z]),
                                            device=device)

        occupancy_map = (pred_x + pred_y + pred_z) / 3
        end = time.time()
        logging.info(f"Prediction obtained and averaged in {end - start}")
        return occupancy_map
