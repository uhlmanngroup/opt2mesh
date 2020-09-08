#! /usr/bin/env python
import numpy as np
import sys
import h5py


def _empirical_crop(volume):
    """
    Limit were determined empirically.

    @param volume:
    @return:
    """
    x_min, x_max = 0, 512
    y_min, y_max = 100, 450
    z_min, z_max = 40, 480

    coords = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

    return volume[x_min:x_max, y_min:y_max, z_min:z_max], coords


if __name__ == "__main__":

    filename = sys.argv[1]
    hdf5_file = h5py.File(filename, "r")
    key = list(hdf5_file.keys())[0]  # == "exported_data"
    data = np.array(hdf5_file[key])
    hdf5_file.close()

    # merging classes of the embryo and the border
    new_data = data.copy()
    new_data[data == 3] = 2

    cropped_data, _ = _empirical_crop(new_data)

    outfile_name = filename.replace(".h5", "_2_classes_cropped.h5")
    hf = h5py.File(outfile_name, "w")
    hf.create_dataset(key, data=cropped_data, chunks=True)
    hf.close()
