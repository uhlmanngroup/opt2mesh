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
        mesh = pymesh.load_mesh(mf)
        v, f = mesh.vertices, mesh.faces
        f = np.asarray(f, dtype=np.int32)
        base_name = mf.split(os.sep)[-1]
        print(f"Testing {base_name}")
        try:
            l = -igl.cotmatrix(v, f)
            print(f" ✅ COT matrix passed")
            m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
            print(f" ✅ Mass matrix passed")
            ind = np.arange(0, v.shape[0], dtype=np.int32)
            g = np.stack([igl.exact_geodesic(v, f, np.array([i], dtype=np.int32), ind) for i in ind])
            print(f" ✅ Geodesic matrix passed")
        except Exception as e:
            print(f" ❌ {base_name} failed")
            print("Exception:", e)
        else:
            print(f"✅ {base_name} passed")
