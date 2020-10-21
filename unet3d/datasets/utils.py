### Code adapted from https://github.com/wolny/pytorch-3dunet, MIT License.

import collections

import numpy as np
import torch
from torch.utils.data import Dataset

from unet3d.utils import get_logger

logger = get_logger("Dataset")


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError


def prediction_collate(batch):
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], tuple) and isinstance(batch[0][1], str):
        return batch[0]
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate([img.ravel() for img in images])
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)
