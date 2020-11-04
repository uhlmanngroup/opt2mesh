import os
import tempfile

import igl
import pytest

from pipeline.unet import UNet3DPipeline, UNetPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")


def test_3dunet():
    """ UNet3DPipeline default settings should run and output the mesh. """
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "3d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNet3DPipeline(model_file=model_file)
        pipeline.run(input_file, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_preprocessed_final_mesh.stl"
        )
        assert os.path.isfile(
            final_mesh_file
        ), "The final mesh is not present in the results."


def test_2dunet():
    """ UNetPipeline default settings should run and output the mesh. """
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "2d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNetPipeline(model_file=model_file)
        pipeline.run(input_file, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_preprocessed_final_mesh.stl"
        )
        assert os.path.isfile(
            final_mesh_file
        ), "The final mesh is not present in the results."


def test_2dunet_save_occupancy_map():
    """ The occupancy map should be saved. """
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "2d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNetPipeline(model_file=model_file, save_occupancy_map=True)
        pipeline.run(input_file, tmp)
        occupancy_map_file = os.path.join(
            tmp, "MNS_M539_105_preprocessed_occupancy_map.tif"
        )
        assert os.path.isfile(
            occupancy_map_file
        ), "The occupancy map is not present after having run the pipeline."


@pytest.mark.parametrize("detail", [3000, 10000, 20000])
def test_2dunet_details(detail):
    """The UNetPipeline should work on various level of details.
    If the detail are integers, output meshes should have less
    faces than the indicated detail."""
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "2d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNetPipeline(model_file=model_file, detail=detail)
        pipeline.run(input_file, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_preprocessed_final_mesh.stl"
        )
        v, f = igl.read_triangle_mesh(final_mesh_file)

        assert len(f) <= detail


@pytest.mark.parametrize("detail", [3000, 10000, 20000])
def test_3dunet_details(detail):
    """The UNet3DPipeline should work on various level of details.
    If the detail are integers, output meshes should have less
    faces than the indicated detail."""
    input_file = os.path.join(test_data_dir, "MNS_M539_105_preprocessed.tif")
    model_file = os.path.join(models_dir, "3d_preprocessed.pytorch")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = UNet3DPipeline(model_file=model_file, detail=detail)
        pipeline.run(input_file, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_preprocessed_final_mesh.stl"
        )
        v, f = igl.read_triangle_mesh(final_mesh_file)

        assert len(f) <= detail
