import os
import sys
from glob import glob

import igl
import pymesh
import numpy as np

__doc__ = "Test whether meshes can be run in the shape analysis pipeline or not."


if __name__ == "__main__":

    mesh_files = glob(os.path.join(sys.argv[1], "**/*.stl"))
    for mf in mesh_files:
        dict_mesh = dict()
        mesh = pymesh.load_mesh(mf)
        v, f = mesh.vertices, mesh.faces
        f = np.asarray(f, dtype=np.int32)
        base_name = mf.split(os.sep)[-1]
        print(f"Testing {base_name}")
        try:
            l = -igl.cotmatrix(v, f)
        except Exception as e:
            print(f" ❌ COT matrix test failed")
            print("Exception:", e)
        else:
            print(f" ✅ COT matrix test passed")

        try:
            m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        except Exception as e:
            print(f" ❌ Mass matrix test failed")
            print("Exception:", e)
        else:
            print(f" ✅ Mass matrix test passed")

        try:
            ind = np.arange(0, v.shape[0], dtype=np.int32)
            g = np.stack(
                [
                    igl.exact_geodesic(v, f, np.array([i], dtype=np.int32), ind)
                    for i in ind
                ]
            )
        except Exception as e:
            print(f" ❌ Geodesic matrix test failed")
            print("Exception:", e)
        else:
            print(f" ✅ Geodesic matrix test passed")
