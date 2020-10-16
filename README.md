# `opt2mesh`: Extract meshes from Optical Projective Tomography scans

### Installation step

 - Clone the repository.
 - Put the models in the `models` folder.
 - Create a conda virtual environment and install the package:
```bash
$ conda env create -f environment.yml
$ conda activate opt2mesh
$ python setup.py install
```

## Usage

Simply run:

```bash
$ opt2mesh opt_scan_as.tif
```

To have all the options, run:

```bash
$ opt2mesh -h
```
