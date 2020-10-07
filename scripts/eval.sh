#!/usr/bin/env bash

# Perform the segmentation evaluation using the metric from

GT_FOLDER=$HOME/gt

PRED_FOLDER=$1

set -e

echo "Computing metric for $PRED_FOLDER"

ev="$/HOME/.scripts/EvaluateSegmentation"

SUFFIX_GT="_clahe_median_denoised_occupancy_map_bin.tif"
SUFFIX_PRED="_clahe_median_denoised_occupancy_map.tif"

val_examples=(
              'MNS_M1054_WS6ga_125'
              'MNS_M173_115'
              'MNS_M188_115'
              'MNS_M525_105'
              'MNS_M539_105'
              'MNS_M566_125'
              )

for val_ex in
do
  echo "Metrics for $PRED_FOLDER/$val_ex$SUFFIX_PRED"
  ev $GT_FOLDER/$val_ex$SUFFIX_GT $PRED_FOLDER/$val_ex$SUFFIX_PRED
done
