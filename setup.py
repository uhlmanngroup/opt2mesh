#! /usr/bin/env python

from setuptools import setup

setup(
    name="opt2mesh",
    version="0.1.0",
    description="Extract meshes for OPT scan.",
    author="Julien Jerphanion",
    author_email="jerphanion@ebi.ac.uk",
    entry_points={"console_scripts": ["opt2mesh = opt2mesh:main"]},
    packages=["unet3d", "unet3d", "unet3d.datasets", "unet", "pipeline"],
)
