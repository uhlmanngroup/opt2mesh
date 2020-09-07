import argparse
import logging
import os

import pymesh
import pyvista as pv
import tetgen

__doc__ = """Tetrahedrisation of a triangular mesh."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument
    parser.add_argument("in_mesh", help="Input triangle mesh (.stl file)")
    parser.add_argument("out_folder", help="General output folder for this run")
    parser.add_argument("--repair", help="Repair the input mesh if needed",
                        action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    pv.set_plot_theme('document')

    filename = args.in_mesh

    # Convert a path like '/path/to/file.name.ext' to 'file.name'
    basename = ".".join(filename.split(os.sep)[-1].split(".")[:-1])

    logging.info(f"Loading mesh {filename}")
    mesh = pymesh.meshio.load_mesh(filename=filename)
    logging.info(f"Vertices: {len(mesh.vertices)}")
    logging.info(f"Faces: {len(mesh.faces)}")

    v = mesh.vertices.copy()
    f = mesh.faces.copy()

    tet = tetgen.TetGen(v, f)

    if args.repair:
        logging.info("Repairing the mesh (pymeshfix)")
        tet.make_manifold(verbose=True)

    # Default parameters from documentation
    tet.tetrahedralize(order=1, minratio=1.5)

    outfile = os.path.join(args.out_folder, basename + '.vtu')
    logging.info(f"Saving the file: {outfile}")
    tet.write(outfile)


if __name__ == "__main__":
    main()
