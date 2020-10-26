import logging
import os
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, erosion, flood_fill

from pipeline import morphsnakes as ms
from pipeline.base import OPT2MeshPipeline

# NOTE: as of now, there's no way to easily access Ilastik from python
# We rely on a simple system call with arguments to the executable but this is cumbersome.
# A python API is being introduced for prediction: https://github.com/ilastik/ilastik/pull/2303
# If this gets in, this would simplify the following implementation _a lot_ with something like this:
#
#     from ilastik.experimental.api import from_project_file
#     pipeline = from_project_file("./MyProjectPixel.ilp")
#     test_res = pipeline.predict(test_arr)
#


class AutoContextACWEPipeline(OPT2MeshPipeline):
    """
    Use AutoContext workflow from Ilastik to segment the object
    the occupancy map (probabilities), then runs ACWE on the
    occupancy map to extract the surface.
    """

    def __init__(
        self,
        # AutoContext specifics
        project,
        # ACWE specifics
        smoothing,
        lambda1,
        lambda2,
        ###
        gradient_direction="descent",
        step_size=1,
        detail=3000,
        iterations=150,
        level=0.5,
        spacing=1,
        save_temp=False,
        segment_occupancy_map=False,
        save_occupancy_map=False,
        align_mesh=False,
        preprocess_opt_scan=False,
    ):
        super().__init__(
            level=level,
            spacing=spacing,
            gradient_direction=gradient_direction,
            step_size=step_size,
            detail=detail,
            save_temp=save_temp,
            segment_occupancy_map=segment_occupancy_map,
            save_occupancy_map=save_occupancy_map,
            align_mesh=align_mesh,
            preprocess_opt_scan=preprocess_opt_scan,
        )

        self.autocontext_pipeline = AutoContextPipeline(
            project=project,
            use_probabilities=True,
            gradient_direction="descent",
            step_size=step_size,
            detail=detail,
            save_temp=save_temp,
            preprocess_opt_scan=preprocess_opt_scan,
        )
        self.iterations = iterations
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _extract_occupancy_map(self, opt2process, base_out_file):
        """
        See AutoContextACWEPipeline docstring.
        """
        logging.info(f"Running Morphological Chan Vese on full")
        start = time.time()
        occupancy_map = self.autocontext_pipeline._extract_occupancy_map(
            opt2process, base_out_file
        )
        end = time.time()
        logging.info(
            f"Done Morphological Chan Vese on full in {(end - start)}s"
        )

        # Initialization of the level-set.
        init_ls = ms.ellipsoid_level_set(occupancy_map.shape)

        logging.info(f"Running Morphological Chan Vese on full")

        start = time.time()
        occupancy_map = ms.morphological_chan_vese(
            occupancy_map,
            init_level_set=init_ls,
            iterations=self.iterations,
            smoothing=self.smoothing,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
        )
        end = time.time()
        logging.info(
            f"Done Morphological Chan Vese on full in {(end - start)}s"
        )

        return occupancy_map


class AutoContextPipeline(OPT2MeshPipeline):
    """
    Use AutoContext workflow from Ilastik to segment the object
    the occupancy map (probabilities).

    The path leading to the Ilastik project needs to be specified
    when constructing the object.

    All the options and some current problems are specified here:
     - https://www.ilastik.org/documentation/basics/headless
    """

    def __init__(
        self,
        # Autocontext specific
        project,
        use_probabilities=True,
        #
        gradient_direction="descent",
        step_size=1,
        detail=3000,
        level=0.5,
        spacing=1,
        save_temp=False,
        segment_occupancy_map=False,
        save_occupancy_map=False,
        align_mesh=False,
        preprocess_opt_scan=False
    ):
        super().__init__(
            level=level,
            spacing=spacing,
            gradient_direction=gradient_direction,
            step_size=step_size,
            detail=detail,
            save_temp=save_temp,
            segment_occupancy_map=segment_occupancy_map,
            save_occupancy_map=save_occupancy_map,
            align_mesh=align_mesh,
            preprocess_opt_scan=preprocess_opt_scan,
        )

        self.project: str = project

        self._use_probabilities = use_probabilities

        if self._use_probabilities:
            # TODO: tune this
            self.level = 0.90

    def _dump_hdf5_on_disk(self, opt2process: np.ndarray, base_out_file: str) -> str:
        """
        Ilastik needs to have a hfd5 file as an input.

        Dump the scan on the disk as a hdf5 file and return the path.
        """
        logging.info(f"Converting to hdf5")
        file_basename = f"{base_out_file}/autocontext/opt2process"
        os.makedirs(f"{base_out_file}/autocontext/", exist_ok=True)
        h5_file = f"{file_basename}.h5"
        hf = h5py.File(h5_file, "w")
        hf.create_dataset("dataset", data=opt2process, chunks=True)
        hf.close()
        logging.info(f"Dumped hdf5 dataset to {h5_file}")

        return h5_file

    def _post_processing(self, interior_segmentation: np.ndarray):
        """
        The result might be noisy after the prediction with Autocontext.
        Hence, we perform some morphological operations on the segmentation
        data.
        """
        improved_seg_data = dilation(
            erosion(
                dilation(gaussian_filter(interior_segmentation, sigma=0.1))
            )
        ).astype(np.uint8)
        improved_seg_data = 255 - flood_fill(improved_seg_data, (1, 1, 1), 255)

        return improved_seg_data

    def _extract_occupancy_map(self, opt2process: np.ndarray, base_out_file: str) -> np.ndarray:

        ilastik_output_folder = f"{base_out_file}/autocontext/predictions/"

        # We use h5 here because it is more memory efficient
        # https://forum.image.sc/t/notable-memory-usage-difference-when-running-ilastik-in-headless-mode-on-different-machines/41144/4
        output_format = "hdf5"
        drange = '"(0,255)"'
        dtype = "uint8"
        output_filename_format = ilastik_output_folder + "{nickname}_pred.h5 "
        in_files = self._dump_hdf5_on_disk(opt2process, base_out_file)

        # Note: one may need some config to have ilastik accessible in PATH
        # NOTE (executable)
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
            command += f"--export_drange={drange} "
            command += f"--pipeline_result_drange={dtype} "
        command += in_files

        logging.info("Lauching Ilastik")
        logging.info("CLI command:")
        logging.info(command)

        # Running the segmentation on ilastik
        out_ilastik = os.popen(command).read()
        logging.info("Ilastik CLI Call standard output:")
        logging.info(out_ilastik)

        segmentation_file = os.path.join(
            ilastik_output_folder, os.listdir(ilastik_output_folder)[0]
        )
        assert segmentation_file.endswith(
            ".h5"
        ), f"Not a correct hdf5 file : {segmentation_file}"

        hf = h5py.File(segmentation_file, "r")
        # We are only interested in the "interior" information.
        # It is the last label, hence the use of "-1"
        noisy_occupancy_map = np.array(hf["exported_data"])[..., -1]
        hf.close()

        if not self._use_probabilities:
            # we work on segmentations which we need to clean
            occupancy_map = self._post_processing(noisy_occupancy_map)
        else:
            occupancy_map = noisy_occupancy_map

        logging.info(f"Done extracting the occupancy map with Autocontext")

        return occupancy_map
