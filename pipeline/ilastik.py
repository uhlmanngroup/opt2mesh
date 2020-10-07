import logging
import os
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.morphology import dilation, erosion

from pipeline import morphsnakes as ms
from pipeline.base import OPT2MeshPipeline
from scripts.preprocessing import to_hdf5, _fill_binary_image


class AutoContextACWEPipeline(OPT2MeshPipeline):
    """
    Use AutoContext to extract the occupancy map (probabilities)
    then runs ACWE on the occupancy map to extract the surface.
    """

    def __init__(
        self,
        # AutoContextSpecific
        project,
        # ACWE specifics
        smoothing,
        lambda1,
        lambda2,
        ###
        gradient_direction="descent",
        step_size=1,
        detail="high",
        iterations=150,
        level=0.5,
        spacing=1,
        save_temp=False,
        on_slices=False,
        n_jobs=-1,
        segment_occupancy_map=False,
        save_occupancy_map=False,
    ):
        super().__init__(
            iterations=iterations,
            level=level,
            spacing=spacing,
            gradient_direction=gradient_direction,
            step_size=step_size,

            detail=detail,
            save_temp=save_temp,
            on_slices=on_slices,
            n_jobs=n_jobs,
            segment_occupancy_map=segment_occupancy_map,
            save_occupancy_map=save_occupancy_map,
        )

        self.autocontext_pipeline = AutoContextPipeline(
            project=project,
            use_probabilities=True,
            gradient_direction="descent",
            step_size=step_size,

            detail=detail,
            save_temp=save_temp,
            on_slices=on_slices,
            n_jobs=n_jobs,
        )
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
        occupancy_map = self.autocontext_pipeline._extract_occupancy_map(
            tif_file, base_out_file
        )
        end = time.time()
        logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")

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
        logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")

        return occupancy_map


class AutoContextPipeline(OPT2MeshPipeline):
    """
    Use ilastik for the segmentation using the headless mode.

    All the options and some current problems are specified here:

    https://www.ilastik.org/documentation/basics/headless
    """

    def __init__(
        self,
        # Autocontext specific
        project,
        use_probabilities=True,
        #
        gradient_direction="descent",
        step_size=1,
        detail="high",
        iterations=150,
        level=0.5,
        spacing=1,
        save_temp=False,
        on_slices=False,
        n_jobs=-1,
        segment_occupancy_map=False,
        save_occupancy_map=False,
    ):
        super().__init__(
            iterations=iterations,
            level=level,
            spacing=spacing,
            gradient_direction=gradient_direction,
            step_size=step_size,
            detail=detail,
            save_temp=save_temp,
            on_slices=on_slices,
            n_jobs=n_jobs,
            segment_occupancy_map=segment_occupancy_map,
            save_occupancy_map=save_occupancy_map,
        )

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
        improved_seg_data = dilation(
            erosion(dilation(gaussian_filter(interior_segmentation, sigma=0.1)))
        ).astype(np.uint8)
        for i in range(improved_seg_data.shape[0]):
            improved_seg_data[i, :, :] = _fill_binary_image(improved_seg_data[i, :, :])

        return improved_seg_data

    def _extract_occupancy_map(self, tif_file, base_out_file):

        ilastik_output_folder = f"{base_out_file}/autocontext/predictions/"

        # We use h5 here because it is more memory efficient
        # https://forum.image.sc/t/notable-memory-usage-difference-when-running-ilastik-in-headless-mode-on-different-machines/41144/4
        output_format = "hdf5"
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
            command += f"--export_drange={drange} "
            command += f"--pipeline_result_drange={dtype} "
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

        if self.save_temp:
            filename = segmentation_file.replace(".h5", f"_occupancy_map.tif")
            logging.info(f"Saving occupancy map in {filename}")
            hf = h5py.File(filename, "w")
            hf.create_dataset("exported_data", data=occupancy_map, chunks=True)
            hf.close()

        logging.info(f"Done extracting the occupancy map with Autocontext")

        return occupancy_map