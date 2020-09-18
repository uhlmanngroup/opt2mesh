import h5py
import numpy as np
import torch

from unet3d.utils import get_logger
from unet3d.utils import remove_halo

logger = get_logger("UNet3DPredictor")


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix="predictions"):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f"{prefix}{i}" for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        out_channels = self.config["model"].get("out_channels")
        if out_channels is None:
            out_channels = self.config["model"]["dt_out_channels"]

        prediction_channel = self.config.get("prediction_channel", None)
        if prediction_channel is not None:
            logger.info(
                f"Using only channel '{prediction_channel}' from the network output"
            )

        device = self.config["device"]
        output_heads = self.config["model"].get("output_heads", 1)

        logger.info(f"Running prediction on {len(self.loader)} batches...")

        # dimensionality of the the output predictions
        volume_shape = self._volume_shape(self.loader.dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(
            f"The shape of the output prediction maps (CDHW): {prediction_maps_shape}"
        )

        patch_halo = self.predictor_config.get("patch_halo", (8, 8, 8))
        self._validate_halo(patch_halo, self.config["loaders"]["test"]["slice_builder"])
        logger.info(f"Using patch_halo: {patch_halo}")

        # create destination H5 file
        h5_output_file = h5py.File(self.output_file, "w")
        # allocate prediction and normalization arrays
        logger.info("Allocating prediction and normalization arrays...")
        prediction_maps, normalization_masks = self._allocate_prediction_maps(
            prediction_maps_shape, output_heads, h5_output_file
        )

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(
                    predictions, prediction_maps, normalization_masks
                ):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f"Saving predictions for slice:{index}...")

                        # remove halo in order to avoid block artifacts in the output probability maps
                        u_prediction, u_index = remove_halo(
                            pred, index, volume_shape, patch_halo
                        )
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results to
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self._get_output_dataset_names(
            output_heads, prefix="predictions"
        )
        for prediction_map, normalization_mask, prediction_dataset in zip(
            prediction_maps, normalization_masks, prediction_datasets
        ):
            prediction_map = prediction_map / normalization_mask

            if self.loader.dataset.mirror_padding is not None:
                z_s, y_s, x_s = [
                    _slice_from_pad(p) for p in self.loader.dataset.mirror_padding
                ]

                logger.info(
                    f"Dataset loaded with mirror padding: {self.loader.dataset.mirror_padding}."
                )

                prediction_map = prediction_map[:, z_s, y_s, x_s]

        return prediction_map

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [
            np.zeros(output_shape, dtype="float32") for _ in range(output_heads)
        ]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [
            np.zeros(output_shape, dtype="uint8") for _ in range(output_heads)
        ]
        return prediction_maps, normalization_masks

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config["patch_shape"]
        stride = slice_builder_config["stride_shape"]

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0
        ), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"
