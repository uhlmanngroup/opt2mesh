import os
import sys

import pandas as pd

__doc__ = "Output LaTeX table and print box-plots of the different statistics using an"

simples = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "Dice Coefficient",
    "Jaccard",
    "Area under ROC Curve",
    "Cohen Kappa",
    "Rand Index",
    "Adjusted Rand Index",
    "Interclass Correlation",
    "Volumetric Similarity Coefficient",
    "Mutual Information",
    "Mahanabolis Distance",
    "Variation of Information",
    "Global Consistency Error",
    "Probabilistic Distance",
]

hd = [
    "Hausdorff Distance (in voxel)",
    # "Average Hausdorff Distance (in voxel)"
]

scores_names = simples + hd


def parse_expe_text(text: str):
    text = text.replace("\t", " ")
    examples_text = text.split("Metrics for")[1:]

    scores_hashmaps = []
    for ex in examples_text:
        lines = ex.split("\n")
        ex_scores = dict()
        ex_scores["Example"] = (
            lines[0]
            .split(os.sep)[-1]
            .replace("_clahe_median_denoised_occupancy_map.tif", "")
        )
        print(ex_scores["Example"])
        for sn in scores_names:
            score_text = list(filter(lambda x: sn in x, lines))[0]
            # SEGVOL  = 7554643       segmented volume (in voxel) → 7554643
            score_value = score_text.split(" = ")[1].split(" ")[0]
            ex_scores[sn] = float(score_value)
        scores_hashmaps.append(ex_scores)

    df = pd.DataFrame(scores_hashmaps).set_index("Example")
    df = df.transpose()
    return df


if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        text = f.read()

    df = parse_expe_text(text)
    df.transpose().boxplot(simples, rot=45)

    # Print LaTeX table with the mean
    mean_df = df.mean(axis=0)
    mean_df.name = "Mean"
    print(
        pd.concat([df, df.mean(axis=1)], axis=1).to_latex(
            index=True, float_format="%.4f"
        )
    )

# Example of output of EvaluateSegmentation
"""
Computing metric for /home/jerphanion/test
Metrics for /home/jerphanion/test/MNS_M1054_WS6ga_125_clahe_median_denoised_occupancy_map.tif
Similarity:
DICE    = 0.904219      Dice Coefficient (F1-Measure)
JACRD   = 0.825183      Jaccard Coefficient
AUC     = 0.914194      Area under ROC Curve
KAPPA   = 0.897913      Cohen Kappa
RNDIND  = 0.976418      Rand Index
ADJRIND = 0.885779      Adjusted Rand Index
ICCORR  = 0.904173      Interclass Correlation
VOLSMTY = 0.908869      Volumetric Similarity Coefficient
MUTINF  = 0.265170      Mutual Information

Distance:
HDRFDST = 21.400935     Hausdorff Distance (in voxel)
------------ Average distance details: -----------------
AVGDST: 5.27971e+06 / 9.06964e+06 = 0.58213
------------ Average distance details end. -----------------
------------ Average distance details: -----------------
AVGDST: 51095.3 / 7.55464e+06 = 0.00676343
------------ Average distance details end. -----------------
AVGDIST = 0.294447      Average Hausdorff Distance (in voxel)
bAVD    = 0.293882      Balanced Average Hausdorff Distance (new metric, publication submitted) (in voxel)
MAHLNBS = 0.063569      Mahanabolis Distance
VARINFO = 0.141838      Variation of Information
GCOERR  = 0.021872      Global Consistency Error
PROBDST = 0.000415      Probabilistic Distance

Classic Measures:
SNSVTY  = 0.828699      Sensitivity (Recall, true positive rate)
SPCFTY  = 0.999689      Specificity (true negative rate)
PRCISON = 0.994885      Precision (Confidence)
FMEASR  = 0.904219      F-Measure
ACURCY  = 0.988067      Accuracy
FALLOUT = 0.000311      Fallout (false positive rate)
TP      = 7515998       true positive (in voxel)
FP      = 38645 false positive (in voxel)
TN      = 124324550     true negative (in voxel)
FN      = 1553638       false negative (in voxel)
REFVOL  = 9069636       reference volume (in voxel)
SEGVOL  = 7554643       segmented volume (in voxel)

Total execution time= 214964 milliseconds


  ---** VISCERAL 2013, www.visceral.eu **---"""
