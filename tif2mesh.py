import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io
from skimage.filters import threshold_otsu

if __name__ == "__main__":

    tiffile = "/tmp/MNS_M897_115.tif"

    img = io.imread(tiffile)[200].astype("float64")

    # img = data.astronaut()
    # img = rgb2gray(img)

    height, width = img.shape

    thresh = threshold_otsu(img)
    bool_img = img > thresh

    s = np.linspace(0, 2*np.pi, 400)
    r = height / 2 * (1 + 0.85 * np.sin(s))
    c = width / 2 * (1 + 0.85 * np.cos(s))
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3),
                           init, alpha=1.5, beta=1,
                           w_line=0, w_edge=50, gamma=0.001,
                           bc=None, max_px_move=1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()
