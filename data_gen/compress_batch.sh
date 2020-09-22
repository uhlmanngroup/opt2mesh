#! /usr/bin/env bash

# Create an archive in the light folder to download
# all but heavy files (like hdf5 or tif)

set -e

BASE_DIR=~/mesh-processing-pipeline/out
JOB_BATCH_NAME=$1

LIGHT_FOLDER=$BASE_DIR/light/$JOB_BATCH_NAME


echo "Creating $LIGHT_FOLDER"
mkdir -p $LIGHT_FOLDER

echo "Copying logs"
find $BASE_DIR/$JOB_BATCH_NAME -name '*.log' -exec cp --parents {} $LIGHT_FOLDER\;

echo "Copying meshes"
find $BASE_DIR/$JOB_BATCH_NAME -name '*final_mesh.stl' -exec cp --parents {} $LIGHT_FOLDER \;

echo "Copying context"
find $BASE_DIR/$JOB_BATCH_NAME -name '*.yml' -exec cp --parents {} $LIGHT_FOLDER \;


echo "Compressing"
tar -czvf $LIGHT_FOLDER.tar.gz $LIGHT_FOLDER

echo "Archive available: $LIGHT_FOLDER.tar.gz!"