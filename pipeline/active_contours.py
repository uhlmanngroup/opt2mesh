import logging
import time

from pipeline import morphsnakes as ms
from pipeline.base import OPT2MeshPipeline


class GACPipeline(OPT2MeshPipeline):
    """
    Use the morphological Geodesic Active Contour [1] to segment the object
    present in the scan.

    The parameters have to be properly calibrated for this method
    to work.

    [1] A Morphological Approach to Curvature-based Evolution of Curves
    and Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez.
    In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
    2014, DOI:10.1109/TPAMI.2013.106
    """

    def __init__(
        self,
        gradient_direction="descent",
        step_size=1,
        detail=6000,
        level=0.5,
        spacing=1,
        save_temp=False,
        segment_occupancy_map=True,
        save_occupancy_map=False,
        align_mesh=False,
        preprocess_opt_scan=False,
        loops_to_remove=None,
        # GAC specifics
        iterations=50,
        n_jobs=-1,
        smoothing=1,
        threshold="auto",
        balloon=1,
        alpha=1000,
        sigma=5,
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
            loops_to_remove=loops_to_remove,
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
    """
    Use the morphological Active Contour Without Edges [1] to segment the
    object present in the scan.

    [1] A Morphological Approach to Curvature-based Evolution of Curves
    and Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez.
    In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
    2014, DOI:10.1109/TPAMI.2013.106
    """

    def __init__(
        self,
        gradient_direction="descent",
        step_size=1,
        detail=6000,
        level=0.5,
        spacing=1,
        save_temp=False,
        segment_occupancy_map=True,
        save_occupancy_map=False,
        align_mesh=False,
        preprocess_opt_scan=False,
        loops_to_remove=None,
        # ACWE specific
        iterations=150,
        smoothing=1,
        lambda1=3,
        lambda2=1,
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
            loops_to_remove=loops_to_remove,
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
        logging.info(
            f"Done Morphological Chan Vese on full in {(end - start)}s"
        )
        del opt2process, init_ls

        return occupancy_map
