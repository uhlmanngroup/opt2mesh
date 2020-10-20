import logging
import time

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
        self.n_jobs: int = n_jobs
        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.alpha = alpha
        self.sigma = sigma

    def _extract_occupancy_map(self, opt2process, base_out_file):
        logging.info(
            f"Starting Morphological Geodesic Active Contour on the full image"
        )
        opt_data = opt2process / 255.0

        gradient_image = ms.inverse_gaussian_gradient(
            opt_data, alpha=self.alpha, sigma=self.sigma
        )

        # Initialization of the level-set.
        init_ls = ms.ellipsoid_level_set(opt_data.shape)

        logging.info(f"Running Morphological Geodesic Active Contour on full")

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
        n_jobs=-1,
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
        self.n_jobs: int = n_jobs
        self.lambda1: int = lambda1
        self.lambda2: int = lambda2
        self.smoothing: int = smoothing

    def _extract_occupancy_map(self, opt2process, base_out_file):
        logging.info(f"Starting Morphological Chan Vese on the full image")

        # Initialization of the level-set.
        init_ls = ms.ellipsoid_level_set(opt2process.shape)

        logging.info(f"Running Morphological Chan Vese on full")

        start = time.time()

        occupancy_map = ms.morphological_chan_vese(
            opt2process,
            init_level_set=init_ls,
            iterations=self.iterations,
            smoothing=self.smoothing,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
        )

        end = time.time()
        logging.info(f"Done Morphological Chan Vese on full in {(end - start)}s")
        del opt2process, init_ls

        return occupancy_map
