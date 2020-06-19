import os
import itertools

import matplotlib.pyplot as plt
from skimage import io, feature
from skimage import filters
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
    fig.savefig(f"{out_folder}/{i}.png")
    plt.close(fig)


if __name__ == "__main__":
    tiffile = "/tmp/MNS_M897_115.tif"

    parallel = Parallel(n_jobs=2)

    opt_data = io.imread(tiffile).astype("float64")

    with parallel:
        sigmas = [10, 15, 20, 25]
        args = itertools.product(list(range(511)), sigmas)

        parallel(delayed(extract_canny)(opt_data[i], i,
                                        gaussian_blur_sigma=1,
                                        sigma=sigma)
                 for (i, sigma) in args)
