import os
import itertools

import matplotlib.pyplot as plt
from skimage import io, feature
from skimage import filters
from skimage.filters import gaussian
from skimage.segmentation import flood, flood_fill
from joblib import Parallel, delayed


def extract_canny(image, i, gaussian_blur_sigma=1, sigma=20):
    slice = filters.gaussian(image, gaussian_blur_sigma)

    print(i)
    print(slice.shape)

    # Compute the Canny filter for two values of sigma
    edges = feature.canny(slice, sigma=sigma)

    # display results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 8),
                                   sharex=True, sharey=True)

    ax1.imshow(slice, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title(f'Slice {i}', fontsize=20)

    ax2.imshow(edges, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma={}$'.format(sigma), fontsize=20)

    out_folder = f"/tmp/canny/{sigma}"

    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(f"{out_folder}/" + f"{i}".zfill(4) + ".png")
    plt.close(fig)


def extract_sobel_flood(image, i, gaussian_blur_sigma=1, tolerance=2):
    slice = filters.gaussian(image, gaussian_blur_sigma)

    print(i)
    h, w = slice.shape

    img_sobel = filters.sobel(slice)
    mask = flood(slice, (0, 0), tolerance=tolerance)

    fig, ax = plt.subplots(ncols=3, figsize=(13, 8))

    ax[0].imshow(slice)
    ax[0].set_title(f'Slice {i}')
    ax[0].axis('off')

    ax[1].imshow(img_sobel)
    ax[1].set_title('Sobel filtered')
    ax[1].axis('off')

    ax[2].imshow(mask, cmap=plt.cm.gray)
    ax[2].set_title(f'Segmented tolerance={tolerance}')
    ax[2].axis('off')

    out_folder = f"/tmp/sobel_flood/{tolerance}"

    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(f"{out_folder}/" + f"{i}".zfill(4) + ".png")
    plt.close(fig)


def canny_filter():
    parallel = Parallel(n_jobs=3)

    sigmas = [5, 10, 15, 20, 25]

    with parallel:
        args = itertools.product(list(range(511)), sigmas)

        parallel(delayed(extract_canny)(opt_data[i], i,
                                        gaussian_blur_sigma=1,
                                        sigma=sigma)
                 for (i, sigma) in args)


def sobel_flood():
    parallel = Parallel(n_jobs=3)

    tolerances = [1.5, 2, 5]

    with parallel:
        args = itertools.product(list(range(511)), tolerances)

        parallel(delayed(extract_sobel_flood)(opt_data[i], i,
                                              gaussian_blur_sigma=1,
                                              tolerance=tolerance)
                 for (i, tolerance) in args)


if __name__ == "__main__":
    tiffile = "/tmp/MNS_M897_115.tif"

    opt_data = io.imread(tiffile).astype("float64")
    sobel_flood()