import logging
import time

import numpy as np
from joblib import Parallel, delayed
from skimage import io

from pipeline import morphsnakes as ms
from pipeline.base import OPT2MeshPipeline


class GACPipeline(OPT2MeshPipeline):
    def __init__(
        self,
        gradient_direction="descent",
        step_size=1,
        detail=3000,
        level=0.5,
        spacing=1,
        save_temp=False,
        # GAC specifics
        iterations=50,
        on_slices=False,
        n_jobs=-1,
        smoothing=1,
        threshold="auto",
        balloon=1,
        alpha=1000,
        sigma=5,
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

        self.iterations: int = iterations
        self.on_slices: bool = on_slices
        self.n_jobs: int = n_jobs
        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.alpha = alpha
        self.sigma = sigma

    def _extract_occupancy_map(self, tif_stack_file, base_out_file):
        logging.info(
            f"Starting Morphological Geodesic Active Contour on the full image"
        )
        logging.info(f"Loading full data")
        opt_data = io.imread(tif_stack_file) / 255.0
        logging.info(f"Loaded full data")

        gradient_image = ms.inverse_gaussian_gradient(
            opt_data, alpha=self.alpha, sigma=self.sigma
        )

        if self.on_slices:
            occupancy_map = np.zeros(gradient_image.shape)

            init_ls = ms.ellipsoid_level_set(gradient_image[0].shape)

            start = time.time()
            logging.info(
                f"Starting Morphological Geodesic Active Contour on slices"
            )
            for i, slice in enumerate(gradient_image):
                logging.info(
                    f"Running Morphological Geodesic Active Contour on slice {i}"
                )

                occupancy_map[
                    i, :, :
                ] = ms.morphological_geodesic_active_contour(
                    slice,
                    iterations=self.iterations,
                    init_level_set=init_ls,
                    smoothing=self.smoothing,
                    threshold=self.threshold,
                    balloon=self.balloon,
                )

            end = time.time()
            logging.info(
                f"Done Morphological Geodesic Active Contour on slices in {(end - start)}s"
            )
        else:
            # Initialization of the level-set.
            init_ls = ms.ellipsoid_level_set(opt_data.shape)

            logging.info(
                f"Running Morphological Geodesic Active Contour on full"
            )

            start = time.time()

            # MorphGAC
            occupancy_map = ms.morphological_geodesic_active_contour(
                gradient_image,
                iterations=self.iterations,
                init_level_set=init_ls,
                smoothing=self.smoothing,
                threshold=self.threshold,
                balloon=self.balloon,
            )
            end = time.time()
            logging.info(
                f"Done Morphological Geodesic Active Contour on full in {(end - start)}s"
            )
            del opt_data, init_ls

        return occupancy_map


class ACWEPipeline(OPT2MeshPipeline):
    def __init__(
        self,
        gradient_direction="descent",
        step_size=1,
        detail=3000,
        level=0.5,
        spacing=1,
        save_temp=False,
        # ACWE specific
        iterations=150,
        on_slices=False,
        n_jobs=-1,
        on_halves=False,
        smoothing=1,
        lambda1=3,
        lambda2=1,
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
        self.iterations: int = iterations
        self.on_slices: bool = on_slices
        self.n_jobs: int = n_jobs
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

            occupancy_map = ms.morphological_chan_vese(
                opt_data,
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

        half_surface = ms.morphological_chan_vese(
            opt_data,
            init_level_set=init_ls,
            iterations=self.iterations,
            smoothing=self.smoothing,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
        )

        end = time.time()
        logging.info(
            f"Done Morphological Chan Vese on {suffix} in {(end - start)}s"
        )

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

        Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(
                self.__acwe_on_one_half(tif_stack_file, base_out_file, suffix)
            )
            for suffix in [
                "x_front",
                "x_back",
                "y_front",
                "y_back",
                "z_front",
                "z_back",
            ]
        )

    def _tif2morphsnakes_slices(self, tif_stack_file):
        """
        Create morphsnakes surfaces on each slices of the cubes independently.

        :param tif_stack_file: path to the TIF stack to process
        """

        opt_data = io.imread(tif_stack_file)
        occupancy_map = np.zeros(opt_data.shape)

        init_ls = ms.ellipsoid_level_set(opt_data[0].shape)

        start = time.time()
        logging.info(f"Starting Morphological Chan Vese on slices")
        for i, slice in enumerate(opt_data):
            logging.info(f"Running Morphological Chan Vese on slice {i}")

            occupancy_map[i, :, :] = ms.morphological_chan_vese(
                slice,
                init_level_set=init_ls,
                iterations=self.iterations,
                smoothing=self.smoothing,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
            )

        end = time.time()
        logging.info(
            f"Done Morphological Chan Vese on slices in {(end - start)}s"
        )

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

        # the cube has a size of (511,512,512)
        x_front_reshaped = np.concatenate(
            (x_front, np.zeros((255, 512, 512), dtype="int8")), axis=0
        )
        x_back_reshaped = np.concatenate(
            (np.zeros((256, 512, 512), dtype="int8"), x_back), axis=0
        )

        logging.info(f"x_front_reshaped.shape: {x_front_reshaped.shape}")
        logging.info(f"x_back_reshaped.shape : {x_back_reshaped.shape}")

        y_front = io.imread(base_out_file + "_y_front.tif")
        y_back = io.imread(base_out_file + "_y_back.tif")

        logging.info(f"y_front.shape         : {y_front.shape}")
        logging.info(f"y_back.shape          : {y_back.shape}")

        y_front_reshaped = np.concatenate(
            (y_front, np.zeros(y_front.shape, dtype="int8")), axis=1
        )
        y_back_reshaped = np.concatenate(
            (np.zeros(y_back.shape, dtype="int8"), y_back), axis=1
        )

        logging.info(
            f"y_front_reshaped.shape: {y_front_reshaped.shape}",
        )
        logging.info(f"y_back_reshaped.shape : {y_front_reshaped.shape}")

        z_front = io.imread(base_out_file + "_z_front.tif")
        z_back = io.imread(base_out_file + "_z_back.tif")

        logging.info(f"z_front.shape         : {z_front.shape}")
        logging.info(f"z_back.shape          : {z_back.shape}")

        z_front_reshaped = np.concatenate(
            (z_front, np.zeros(z_front.shape, dtype="uint8")), axis=2
        )
        z_back_reshaped = np.concatenate(
            (np.zeros(z_back.shape, dtype="uint8"), z_back), axis=2
        )

        logging.info(f"z_front_reshaped.shape: {z_front_reshaped.shape}")
        logging.info(f"z_back_reshaped.shape : {z_back_reshaped.shape}")

        # The full segmentation surface
        occupancy_map = (
            x_front_reshaped
            + x_back_reshaped
            + y_back_reshaped
            + y_front_reshaped
            + z_back_reshaped
            + z_front_reshaped
        ).clip(0, 1)

        if self.save_temp:
            io.imsave(
                base_out_file + "_x_front_reshaped.tif", x_front_reshaped
            )
            io.imsave(base_out_file + "_x_back_reshaped.tif", x_back_reshaped)

            io.imsave(
                base_out_file + "_y_front_reshaped.tif", y_front_reshaped
            )
            io.imsave(base_out_file + "_y_back_reshaped.tif", y_back_reshaped)

            io.imsave(
                base_out_file + "_z_front_reshaped.tif", z_front_reshaped
            )
            io.imsave(base_out_file + "_z_back_reshaped.tif", z_back_reshaped)

        return occupancy_map
