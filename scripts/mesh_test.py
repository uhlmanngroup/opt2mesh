import os
import sys
from glob import glob

import igl
import pymesh
import numpy as np

__doc__ = "Test whether meshes can be run in the pipeline or not."


if __name__ == "__main__":

    mesh_files = glob(os.path.join(sys.argv[1], "*.stl"))

    for mf in mesh_files:
        mesh = pymesh.load_mesh(mf)
        v = mesh.vertices
        f = mesh.faces
        base_name = mf.split(os.sep)[-1]
        try:
            l = -igl.cotmatrix(v, f)
            m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
            ind = np.arange(0, v.shape[0])
            # g = np.stack([igl.exact_geodesic(v, f, np.array([i], dtype=c_int), ind) for i in ind])
        except Exception as e:
            print(f"❌ {base_name} failed")
            print("Exception:", e)
        else:
            print(f"✅ {base_name} passed")
