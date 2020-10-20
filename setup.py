#! /usr/bin/env python
import glob
import os

from setuptools import setup

setup(
    name="opt2mesh",
    version="0.1.0",
    description="Extract meshes for OPT scan.",
    author="Julien Jerphanion",
    author_email="jerphanion@ebi.ac.uk",
    entry_points={"console_scripts": ["opt2mesh = opt2mesh:main"]},
    packages=["unet3d", "unet3d.datasets", "unet", "pipeline"],
    package_data={"unet3d": ["*.yml"]},
    data_files=[("models", glob.glob(os.path.join("models", "*.pytorch")))],
)