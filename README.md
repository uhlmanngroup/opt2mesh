# `opt2mesh`: Extract meshes from Optical Projective Tomography scans

### Installation step

 - Clone the repository.
 - Put the models in the `models` folder
    - Some models which have been pretrained (from OPT scans of embryos for the
`UNetPipeline` and the `UNet3DPipeline` are available here: https://oc.ebi.ac.uk/s/mVuDrBtROqTadhc)
 - Create a conda virtual environment and install the package:
```bash
$ conda env create -f environment.yml
$ conda activate opt2mesh
$ python setup.py install
```

## CLI Usage

You can access the method
Simply run:

```bash
$ opt2mesh opt_scan.tif /path/to/result/folder
```

This will create a subfolder containing:
 - the mesh extracted (`opt_scan.stl`)
 - logs of the job ran
 - information about the mesh correctness and quality

You can store the result of the segmentation by using:

```bash
$ opt2mesh --save_occupancy_map opt_scan.tif
```


All the options are available using the help flag:
```bash
$ opt2mesh -h
```

⚠ If you get:
```
ModuleNotFoundError: No module named 'opt2mesh'
```

when running `opt2mesh` directly, just use:

```bash
$ python opt2mesh
```

## Segmentation methods

Several methods for segmentation are available to identify the object in the image.
See the `--method` flag:
 - `acwe` uses morphological variant of Active Contours Without Edge.
 - `gac` uses morphological variant of Geodesic Active Contours.
 - `2d_unet` uses a 2D U-Net trained on this implementation.
 - `3d_unet` uses a 3D U-Net trained on this implementation. You definitely require a GPU
for this method.
 - `direct` performs a direct meshing on a already OPT scan. This is the default one method.

Each of those methods has its own set of parameters which can be set on the command line.
See the help for more detail:

```bash
$ opt2mesh -h
```

### Introducing a custom segmentation method

You can introduce your own segmentation method in the pipeline.

To do this, you have to extend the base class `OPT2MeshPipeline` and to define the
`_extract_occupancy_map` method.

