#! /usr/bin/env python

import argparse
import pandas as pd


__doc__ = "Simple CSV to LaTeX renderer for mesh statistics"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", help="CSV files containing mesh statistics")

    args = parser.parse_args()

    df = pd.read_csv(args.stats)

    test_ex_mapping = {
        "MNS_M1054_WS6ga_125_clahe_median_denoised_occupancy_map_final_mesh.stl": "A",
        "MNS_M173_115_clahe_median_denoised_occupancy_map_final_mesh.stl": "B",
        "MNS_M566_125_clahe_median_denoised_occupancy_map_final_mesh.stl": "C",
        "MNS_M525_105_clahe_median_denoised_occupancy_map_final_mesh.stl": "D",
        "MNS_M188_115_clahe_median_denoised_occupancy_map_final_mesh.stl": "E",
        "MNS_M539_105_clahe_median_denoised_occupancy_map_final_mesh.stl": "F",
    }
    df.mesh_name = df.mesh_name.map(test_ex_mapping)
    old_cols = df.keys()

    new_cols = list(map(lambda x: x.replace("_", " ").capitalize(), df.keys()))

    df = df.rename(columns=dict(zip(old_cols, new_cols)))

    # print(pd.concat([df, df.mean(axis=1)], axis=1).to_latex(index=True, float_format="%.4f"))
    big_values_col = [
        "Total volume",
        "Number of vertices",
        "Number of faces",
        "Number of edges",
        "Total area",
        "Maximum aspect ratio",
        "Minimum altitude",
    ]

    # df.drop(columns=big_values_col).boxplot(rot=90)
    # df[big_values_col].boxplot(rot=90)

    out_df = df.transpose()
    out_df.columns = out_df.loc["Mesh name"]
    out_df = out_df.drop("Mesh name")
    print(
        out_df.rename(
            columns=dict(zip(out_df.keys(), ["A", "B", "C", "D", "E", "F"]))
        ).to_latex()
    )
