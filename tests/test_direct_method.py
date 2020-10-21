import os
import tempfile

from pipeline.base import DirectMeshingPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

def test_dummy():
    print("pass")


def test_direct_method():
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = DirectMeshingPipeline()
        pipeline.run(input_file_binary, tmp)