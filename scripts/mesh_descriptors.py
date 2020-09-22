#! /usr/bin/env python

import os
import argparse

import numpy as np
import pymesh
from scipy.sparse.linalg import eigsh

import igl


__doc__ = "Some test with mesh descriptors"


def laplace_eigen_decomposition(v, f, k):
    l = -igl.cotmatrix(v, f)
    print("a")
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    print("b")
    print(l.shape, m.shape)
    evals, evecs = eigsh(A=l, k=k, M=m, which="SM")
    evecs /= np.sqrt(np.sum(m.dot(evecs ** 2), axis=0, keepdims=True))
    return [evals, evecs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_mesh", help="The mesh to repair")

    args = parser.parse_args()

    print(args.input_mesh)

    mesh = pymesh.load_mesh(args.input_mesh)

    v, f = mesh.vertices, mesh.faces
    print(len(v), len(f))

    evals, evecs = laplace_eigen_decomposition(v, f, k=3)

    basename = args.input_mesh.split(os.sep)[-1]
    print(f"Laplace Eigenvalues for {basename}")
    print(evals)
    print(f"Laplace Eigenvectors for {basename}")
    print(evecs)
    print()