#!/bin/bash

# Configuration
CORPUS="dtfit"

MODEL=$1 # "EleutherAI/pythia-70m-deduped", "OLMo-1B-hf", "google/flan-t5-small" 
REVISION=$2 # "main" or see intermediate checkpoints for models
QUANTIZATION=${3:-"full"}  # "4bit" or "8bit", optional

SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
RESULTDIR="results/exp2_word-comparison"
DATAFILE="datasets/exp2/${CORPUS}/corpus.csv"

# # Define variable-dependent file/folder names
# OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}_"

if [ "$QUANTIZATION" != "full" ]; then
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}"
else
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${REVISION}"
fi

# Ensure the result directory exists
mkdir -p $RESULTDIR

# Call the Python script with the necessary arguments
python run_exp2_word-comparison.py $MODEL $REVISION $QUANTIZATION $DATAFILE $OUTFILE

echo "All tasks completed successfully. Files stored in'"${RESULTDIR}"'"
