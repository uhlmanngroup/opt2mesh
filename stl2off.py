import argparse
import logging
import os

import pymesh
import pyvista as pv
import tetgen

__doc__ = """STL to OFF converter"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("in_mesh", help="Input triangle mesh (.stl file)")
    parser.add_argument("out_folder", help="General output folder for this run")

    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    filename = args.in_mesh

    logging.info(f"Loading mesh: {filename}")
    mesh = pymesh.meshio.load_mesh(filename=filename)
    logging.info(f"Vertices: {len(mesh.vertices)}")
    logging.info(f"Faces: {len(mesh.faces)}")

    outfilename = filename.replace(".stl", ".off")

    logging.info(f"Saving mesh: {outfilename}")
    pymesh.meshio.save_mesh(outfilename, mesh)


if __name__ == "__main__":
    main()
