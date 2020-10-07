#! /usr/bin/env python

import sys

import pandas as pd
import matplotlib.pyplot as plt

import os

selected_examples = {
    "MNS_M322_1_clahe_median_denoised_final_mesh.stl",
    "MNS_M395_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M432_105_clahe_median_denoised_final_mesh.stl",
    "MNS_M522_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M525_105_clahe_median_denoised_final_mesh.stl",
    "MNS_M526_105_clahe_median_denoised_final_mesh.stl",
    "MNS_M539_105_clahe_median_denoised_final_mesh.stl",
    "MNS_M566_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M581_1_clahe_median_denoised_final_mesh.stl",
    "MNS_M583_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M583a_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M590_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M607_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M612_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M620_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M631_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M638_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M693_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M699_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M703_125refocused_clahe_median_denoised_final_mesh.stl",
    "MNS_M714_115_clahe_median_denoised_final_mesh.stl",
    "MNS_M738_125_clahe_median_denoised_final_mesh.stl",
    "MNS_M744_115_clahe_median_denoised_final_mesh.stl",
}


if __name__ == "__main__":

    files = os.listdir(sys.argv[1])

    dfs = {}
    for file in files:
        t = file.replace(".csv", "")
        dfs[t] = pd.read_csv(file).set_index("mesh_name").loc[selected_examples]

    genuses = pd.DataFrame()
    for k, df in dfs.items():
        genuses[k] = df.genus

    print(genuses.mean())

    genuses.boxplot(rot=45, fontsize=15)
    plt.show()
