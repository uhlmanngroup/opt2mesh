#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

"""
Settings of the project.
"""

# FOLDERS
ROOT = Path(__file__).parents[2]

# Data given
DATA_FOLDER = ROOT / "data"
OUT_FOLDER = ROOT / "out"

# Matplotlib figure size
DEFAULT_FIG_SIZE = (30, 15)

# To prevent infinite values or nan's
eps = 1e-8