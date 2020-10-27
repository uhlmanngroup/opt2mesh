import glob
import os
import tempfile

import igl
import pytest

from pipeline.base import DirectMeshingPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")


def test_direct_meshing():
    """ DirectMeshingPipeline default settings should run and output the mesh. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline()
        pipeline.run(input_file_binary, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_binary_final_mesh.stl"
        )
        assert os.path.isfile(
            final_mesh_file
        ), "The final mesh is not present in the results."


def test_direct_meshing_save_temp():
    """Temporary artifacts should be saved. In particular the raw extracted
    mesh should and its connected components should be present."""
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline(save_temp=True)
        pipeline.run(input_file_binary, tmp)
        extracted_mesh_file = os.path.join(
            tmp, "MNS_M539_105_binary_extracted_mesh.stl"
        )
        assert os.path.isfile(
            extracted_mesh_file
        ), "The extracted mesh is not present after having run the pipeline."
        connected_components_files = glob.glob(
            os.path.join(tmp, "MNS_M539_105_binary_extracted_mesh_*.stl")
        )
        assert (
            len(connected_components_files) == 36
        ), "There must be 36 connected components."


def test_direct_meshing_save_occupancy_map():
    """ The occupancy map should be saved. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline(save_occupancy_map=True)
        pipeline.run(input_file_binary, tmp)
        occupancy_map_file = os.path.join(
            tmp, "MNS_M539_105_binary_occupancy_map.tif"
        )
        assert os.path.isfile(
            occupancy_map_file
        ), "The occupancy map is not present after having run the pipeline."


@pytest.mark.parametrize("detail", ["low", "normal", "high", "original"])
def test_direct_meshing_detail(detail):
    """ The direct meshing pipeline should work on various level of details. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline(detail=detail)
        pipeline.run(input_file_binary, tmp)


@pytest.mark.parametrize("detail", [3000, 10000, 20000])
def test_direct_meshing_detail_str(detail):
    """The direct meshing pipeline should work on various level of details.
    If the detail are integers, output meshes should have less
    faces than the indicated detail."""
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline(detail=detail)
        pipeline.run(input_file_binary, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_binary_final_mesh.stl"
        )
        v, f = igl.read_triangle_mesh(final_mesh_file)

        assert len(f) <= detail
