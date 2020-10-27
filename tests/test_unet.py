import os
import tempfile

from pipeline.unet import UNet3DPipeline, UNetPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")



def test_3dunet():
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "3d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNet3DPipeline(model_file=model_file)
        pipeline.run(input_file, tmp)


def test_2dunet():
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "2d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNetPipeline(model_file=model_file)
        pipeline.run(input_file, tmp)
