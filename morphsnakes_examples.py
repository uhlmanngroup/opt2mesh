import itertools
import os
import logging

import numpy as np
from imageio import imread
import matplotlib
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage import io

import morphsnakes as ms

# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.

    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.0001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def visual_callback_3d(fig=None, plot_each=1):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 3D images.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    plot_each : positive integer
        The plot will be updated once every `plot_each` calls to the callback
        function.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """

    from mpl_toolkits.mplot3d import Axes3D
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(0.001)

    counter = [-1]

    def callback(levelset):

        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        if ax.collections:
            del ax.collections[0]

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.pause(0.1)

    return callback


def extract_morphsnakes(img, i):
    print(i)
    logging.info('Running: example_coins (MorphGAC)...')

    # g(I)
    gimg = ms.inverse_gaussian_gradient(img)

    # Manual initialization of the level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[20:-20, 20:-20] = 1

    # Callback for visual plotting
    # callback = visual_callback_2d(img)

    # MorphGAC.
    res = ms.morphological_geodesic_active_contour(gimg,
                                                   iterations=500,
                                                   init_level_set=init_ls,
                                                   smoothing=1,
                                                   threshold=0.69,
                                                   balloon=-1)
    out_folder = f"/tmp/morphsnakes/"

    os.makedirs(out_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))

    ax.imshow(res)
    ax.set_title(f'Slice {i}')
    ax.axis('off')

    fig.savefig(f"{out_folder}/" + f"{i}".zfill(4) + ".png")
    plt.close(fig)

    return res


def morphosnakes(opt_data):
    parallel = Parallel(n_jobs=3)

    with parallel:
        parallel(delayed(extract_morphsnakes)(opt_data[i], i) for i in range(70, 511))


if __name__ == '__main__':
    # Uncomment the following line to see a 3D example
    # This is skipped by default since mplot3d is VERY slow plotting 3d meshes
    # example_confocal3d()
    logging.basicConfig(level=logging.DEBUG)
    from guppy import hpy

    h = hpy()
    print("Before loading the data")
    print(h.heap())

    tiffile = "/tmp/MNS_M897_115.tif"

    half_size = 256

    opt_data = io.imread(tiffile)[:, :half_size, :]
    print("After loading the data")
    print(h.heap())

    # Initialization of the level-set.
    init_ls = ms.circle_level_set(opt_data.shape)

    # Morphological Chan-Vese (or ACWE)
    half_mesh = ms.morphological_chan_vese(opt_data, iterations=150,
                                           init_level_set=init_ls,
                                           smoothing=1, lambda1=1, lambda2=2)

    io.imsave(tiffile.replace(".tif", "_right.tif"), half_mesh)
