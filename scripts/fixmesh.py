#! /usr/bin/env python

import argparse
import os
from pprint import pprint

import pymesh
import pymeshfix
import numpy as np

__doc__ = "Manual post-processing of meshes with defect"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_mesh", help="The mesh to repair")
    parser.add_argument("output_folder", help="The output folder")

    args = parser.parse_args()

    mesh = pymesh.load_mesh(args.input_mesh)

    v = np.copy(mesh.vertices)
    f = np.copy(mesh.faces)

    # Create object from vertex and face arrays
    print(f"Fixing mesh {args.input_mesh}")
    meshfix = pymeshfix.MeshFix(v, f)

    # Repair input mesh
    meshfix.repair()

    # Access the repaired mesh with vtk
    fixed_mesh = meshfix.mesh
    os.makedirs(args.output_folder, exist_ok=True)

    base_name_file = args.input_mesh.split(os.sep)[-1]
    out_file = os.path.join(args.output_folder, base_name_file)
    print(f"Saving fixed mesh in {out_file}")
    pymesh.save_mesh_raw(out_file, meshfix.v, meshfix.f)
    cout_mesh_statistics = (
        os.popen(f"mesh_statistics -i {out_file}").read().split("\n")[:-1]
    )
    # cout_mesh statistics is a list of string of the form:
    # Name of statistic: value
    # here we parse it to get a dictionary of the item:
    #  {"name_of_statistic": value, …}
    mesh_statistics = {
        t[0].strip().lower().replace(" ", "_"): float(t[1])
        for t in map(lambda x: x.split(":"), cout_mesh_statistics)
    }

    pprint(mesh_statistics)
