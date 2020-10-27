import os
import tempfile

from pipeline.base import DirectMeshingPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")


def test_direct_meshing():
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline()
        pipeline.run(input_file_binary, tmp)


