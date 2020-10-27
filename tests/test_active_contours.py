import os
import tempfile

import pytest

from pipeline.active_contours import GACPipeline, ACWEPipeline

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
models_dir = os.path.join(root_dir, "models")


@pytest.mark.parametrize("method", [GACPipeline, ACWEPipeline])
def test_active_contour_pipeline(method):
    """ Active contour methods run and output the mesh. """
    input_file_binary = os.path.join(test_data_dir, "MNS_M539_105_binary.tif")
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = method()
        pipeline.run(input_file_binary, tmp)
        final_mesh_file = os.path.join(
            tmp, "MNS_M539_105_binary_final_mesh.tif"
        )
        assert os.path.isfile(
            final_mesh_file
        ), "The final mesh is not present in the results."
