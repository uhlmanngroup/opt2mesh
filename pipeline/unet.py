import logging
import os
import shutil
import time

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from skimage import io
from torch.nn import functional as F
from torchvision import transforms

from pipeline.base import OPT2MeshPipeline
from unet import UNet
from unet.dataset import BasicDataset
from unet3d.datasets.hdf5 import get_test_loaders
from unet3d.model import UNet3D
from unet3d.predictor import StandardPredictor


class UNetPipeline(OPT2MeshPipeline):
    """
    Use a 2D UNet to get occupancy map slices on the 3 different axes.
    Predictions are stacked together to get occupancy maps and are then
    averaged to get a better estimated occupancy map.

    Code adapted from:
     - https://github.com/milesial/Pytorch-UNet

    """

    def __init__(
        self,
        # UNet (2D) specifics
        model_file,
        scale_factor=0.5,
        bilinear=False,
        ###
        level=0.5,
        gradient_direction="descent",
        step_size=1,
        detail=3000,
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

        self.model_file = model_file
        self.scale_factor = scale_factor
        self.bilinear = bilinear

    def _predict(self, net, full_img, device):
        net.eval()

        img = torch.from_numpy(
            BasicDataset.preprocess(full_img, self.scale_factor)
        )

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
                    transforms.ToTensor(),
                ]
            )

            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask

    def _extract_occupancy_map(self, opt2process, base_out_file):
        logging.info(f"Running 2D UNet on the 3 axis")
        start = time.time()

        first, last = 0, 511
        opt2process = opt2process[first:last, first:last, first:last]

        pred_x = np.zeros(opt2process.shape)
        pred_y = np.zeros(opt2process.shape)
        pred_z = np.zeros(opt2process.shape)

        h, w, d = opt2process.shape

        net = UNet(n_channels=1, n_classes=1, bilinear=self.bilinear)

        logging.info("Loading model {}".format(self.model_file))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {device}")
        net.to(device=device)
        net.load_state_dict(torch.load(self.model_file, map_location=device))

        logging.info(f"Prediction w.r.t axis x")
        for x in range(h):
            logging.info(f"Slice x: {x}/{h}")
            pred_x[x, :, :] = self._predict(
                net=net, full_img=Image.fromarray(opt2process[x, :, :]), device=device
            )

        logging.info(f"Prediction w.r.t axis y")
        for y in range(w):
            logging.info(f"Slice y: {y}/{w}")
            pred_y[:, y, :] = self._predict(
                net=net, full_img=Image.fromarray(opt2process[:, y, :]), device=device
            )

        logging.info(f"Prediction w.r.t axis z")
        for z in range(d):
            logging.info(f"Slice z: {z}/{d}")
            pred_z[:, :, z] = self._predict(
                net=net, full_img=Image.fromarray(opt2process[:, :, z]), device=device
            )

        occupancy_map = (pred_x + pred_y + pred_z) / 3
        end = time.time()
        logging.info(f"Prediction obtained and averaged in {end - start}")
        return occupancy_map


class UNet3DPipeline(OPT2MeshPipeline):
    """
    Use a 3D UNet to get occupancy map.
    """

    def __init__(
        self,
        # UNet (3D) specifics
        model_file,
        config_file=None,
        patch_halo=None,
        stride_shape=None,
        f_maps=None,
        ###
        level=0.5,
        gradient_direction="descent",
        step_size=1,
        detail=3000,
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

        self.model_file = model_file

        if config_file is None:
            global_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.path.pardir
            )
            config_file = os.path.join(global_dir, "unet3d", "config.yml")

        config = yaml.safe_load(open(config_file, "r"))

        # Configuration overriding
        if f_maps is not None:
            logging.info(f"Override Configuration: use f_maps={f_maps} ")
            config["model"]["f_maps"] = f_maps

        if patch_halo is not None:
            t_patch = (patch_halo, patch_halo, patch_halo)
            logging.info(f"Override Configuration: use patch_halo={t_patch} ")
            config["predictor"]["patch_halo"] = t_patch

        if stride_shape is not None:
            t_stride = (stride_shape, stride_shape, stride_shape)
            logging.info(
                f"Override Configuration: use stride_shape={t_stride} "
            )
            config["loaders"]["test"]["slice_builder"][
                "stride_shape"
            ] = t_stride

        # Get a device to train on
        device_str = config.get("device", None)
        if device_str is not None:
            logging.info(f"Device specified in config: '{device_str}'")
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                logging.info("CUDA not available, using CPU")
                device_str = "cpu"
        else:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using '{device_str}' device")

        device = torch.device(device_str)
        config["device"] = device

        self.config = config

    def __load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Loads model and training parameters from a given checkpoint_path
        If optimizer is provided, loads optimizer's state_dict of as well.

        Args:
            checkpoint_path (string): path to the checkpoint to be loaded
            model (torch.nn.Module): model into which the parameters are to be copied
            optimizer (torch.optim.Optimizer) optional: optimizer instance into
                which the parameters are to be copied

        Returns:
            state
        """
        if not os.path.exists(checkpoint_path):
            raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        return state

    def __get_output_file(
        self, dataset, suffix="_predictions", output_dir=None
    ):
        input_dir, file_name = os.path.split(dataset.file_path)
        if output_dir is None:
            output_dir = input_dir
        output_file = os.path.join(
            output_dir, os.path.splitext(file_name)[0] + suffix + ".h5"
        )
        return output_file

    def _extract_occupancy_map(self, opt2process, base_out_file):
        logging.info(f"Running 3D UNet")

        first, last = 0, 511
        opt2process = opt2process[first:last, first:last, first:last]

        h5_dir = f"{base_out_file}/h5"
        os.makedirs(h5_dir, exist_ok=True)
        h5_file = f"{h5_dir}/opt2process.h5"

        hf = h5py.File(h5_file, "w")
        logging.info(f"Converting to {h5_file}")
        data_set_name = self.config["loaders"]["raw_internal_path"]
        hf.create_dataset(data_set_name, data=opt2process, chunks=True)
        hf.close()

        logging.info(f"Adding {h5_dir} as the file path for predictions")
        self.config["loaders"]["test"]["file_paths"] = [h5_dir]

        # Create the model
        model = UNet3D(**self.config["model"])

        # Load model state
        logging.info(f"Loading model from {self.model_file}...")
        self.__load_checkpoint(self.model_file, model)

        device = self.config["device"]
        if torch.cuda.device_count() > 1 and not device.type == "cpu":
            model = torch.nn.DataParallel(model)
            logging.info(
                f"Using {torch.cuda.device_count()} GPUs for prediction"
            )

        logging.info(f"Sending the model to '{device}'")
        model = model.to(device)

        predictions_output_dir = f"{base_out_file}/predictions"

        os.makedirs(predictions_output_dir, exist_ok=True)
        logging.info(f"Saving predictions to: {predictions_output_dir}")

        for test_loader in get_test_loaders(self.config):
            logging.info(f"Processing '{test_loader.dataset.file_path}'...")

            output_file = self.__get_output_file(
                dataset=test_loader.dataset, output_dir=predictions_output_dir
            )

            predictor_config = self.config.get("predictor", {})
            predictor = StandardPredictor(
                model,
                test_loader,
                output_file,
                self.config,
                **predictor_config,
            )

            # Run the model prediction on the entire dataset
            occupancy_map = predictor.predict()

        if len(occupancy_map.shape) == 4:
            occupancy_map = occupancy_map[0, :, :, :]

        logging.info(f"Removing temporary directory '{h5_dir}'")
        shutil.rmtree(h5_dir)

        return occupancy_map
