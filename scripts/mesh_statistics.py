#! /usr/bin/env python

import argparse
import os
import pandas as pd
from pathlib import Path
from pprint import pprint
from joblib import Parallel, delayed

__doc__ = "Compute statistics of a set of meshes and stores them in a CSV file "


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("meshes_folder", help="Folder containing STL files of the batch of jobs.")
    parser.add_argument("out_folder", help="Out folder for this run")

    return parser.parse_args()


def get_statistics(mesh_file):
    """
    Get statistics using the executable compiled with CGAL.
    """

    cout_mesh_statistics = (
        os.popen(f"mesh_statistics -i {mesh_file}").read().split("\n")[:-1]
    )
    # cout_mesh statistics is a list of string of the form:
    # Name of statistic: value
    # here we parse it to get a dictionary of the item:
    #  {"name_of_statistic": value, …}
    mesh_statistics = {
        t[0].strip().lower().replace(" ", "_"): float(t[1])
        for t in map(lambda x: x.split(":"), cout_mesh_statistics)
    }

    mesh_statistics["mesh_name"] = str(mesh_file).split(os.sep)[-1]

    return mesh_statistics


if __name__ == "__main__":
    args = parse_args()

    mesh_files = sorted(list(Path(args.meshes_folder).rglob('*.stl')))

    os.makedirs(args.out_folder, exist_ok=True)

    print(len(mesh_files), "files")

    list_dicts = []

    for f in mesh_files:
        print(f)
        mesh_statistics = get_statistics(f)
        list_dicts.append(mesh_statistics)
        pprint(mesh_statistics)

    basename = args.meshes_folder.split(os.sep)[-1]

    df = pd.DataFrame(list_dicts).set_index('mesh_name')
    out_file = os.path.join(args.out_folder, f"{basename}.csv")
    df.to_csv(out_file)
    print(df)
    print(f"Mesh statistics of {args.meshes_folder} saved in", out_file)

