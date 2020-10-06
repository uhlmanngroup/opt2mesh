#! /usr/bin/env python

import argparse
import glob

from skimage import io
import os

import matplotlib.pyplot as plt
import numpy as np


def show_images(images, cols=1, titles=None, top_title=None, fn=None):
    """Display a list of images in a single figure with matplotlib.

    Taken from:
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    if fn is not None:
        plt.ioff()
        plt.close()
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        ax = plt.gca()
        # if image.ndim == 2:
        #     plt.gray()
        plt.imshow(image, cmap=plt.get_cmap('viridis'))
        a.set_title(title)
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.subplots_adjust(wspace=None, hspace=None)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0)
    fig.tight_layout()
    plt.axis('off')
    if top_title is not None:
        fig.suptitle(top_title, fontsize=16)
    plt.grid(b=None)
    plt.autoscale(tight=True)
    if fn:
        plt.savefig(fn, dpi=10)
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preds", help="2d binary predictions")
    parser.add_argument("outfolder", help="2d binary predictions")

    args = parser.parse_args()

    cols = 4

    batch = os.path.dirname(args.preds).split(os.sep)[-1]
    os.makedirs(args.outfolder, exist_ok=True)

    preds = sorted(glob.glob(os.path.join(args.preds, "*.tif")))
    for ex_f in preds:
        ex_2db = io.imread(ex_f)
        ex_name = ex_f.split(os.sep)[-1].replace("_clahe_median_denoised_occupancy_map.tif", "")
        title = f'{batch}: {ex_name}'

        print(title, "xy slice")
        slices = np.linspace(start=50, stop=450, num=cols * 8, dtype=np.uint)
        titles = [f"Slice {i}" for i in slices]
        fn = os.path.join(args.outfolder, f"{batch}_{ex_name}_z.png")
        sss = [ex_2db[i, :, :] for i in slices]
        show_images(sss, cols=cols, titles=titles, top_title=f"{title} xy axis", fn=fn)

        print(title, "xz slice")
        slices = np.linspace(start=100, stop=400, num=cols * 8, dtype=np.uint)
        titles = [f"Slice {i}" for i in slices]
        fn = os.path.join(args.outfolder, f"{batch}_{ex_name}_y.png")
        sss = [ex_2db[:, i, :] for i in slices]
        show_images(sss, cols=cols, titles=titles, top_title=f"{title} xz axis", fn=fn)

        print(title, "yz slice")
        slices = np.linspace(start=100, stop=400, num=cols * 8, dtype=np.uint)
        titles = [f"Slice {i}" for i in slices]
        fn = os.path.join(args.outfolder, f"{batch}_{ex_name}_x.png")
        sss = [ex_2db[:, :, i] for i in slices]
        show_images(sss, cols=cols, titles=titles, top_title=f"{title} yz axis", fn=fn)

