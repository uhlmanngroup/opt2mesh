import os
import tempfile

from pipeline.base import DirectMeshingPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")


def test_direct_meshing():
    """ DirectMeshingPipeline default settings should run. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline()
        pipeline.run(input_file_binary, tmp)


def test_direct_meshing_save_temp():
    """ Temporary artifacts should be saved. """
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


def test_direct_meshing_save_occupancy_map():
    """ The occupancy map should be saved. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline(save_occupancy_map=True)
        pipeline.run(input_file_binary, tmp)
        occupancy_map_file = os.path.join(
            tmp, "MNS_M539_105_occupancy_map.tif"
        )
        assert os.path.isfile(
            occupancy_map_file
        ), "The occupancy map is not present after having run the pipeline."
