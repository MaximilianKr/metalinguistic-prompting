#!/bin/bash

CORPUS="dtfit"

# Required, "EleutherAI/pythia-70m-deduped", "allenai/OLMo-1B-hf", "google/flan-t5-small"
MODEL=$1

# Optional, default "main" / see intermediate checkpoints for models
REVISION=${2:-"main"}

# Optional, default "full" precision, possible "4bit" or "8bit"
QUANTIZATION=${3:-"full"}

SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
RESULTDIR="results/exp2_word-comparison"
DATAFILE="datasets/exp2/${CORPUS}/corpus.csv"

if [ "$QUANTIZATION" != "full" ]; then
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}"
else
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${REVISION}"
fi

mkdir -p $RESULTDIR

python run_exp2_word-comparison.py $MODEL $REVISION $QUANTIZATION $DATAFILE $OUTFILE

echo "All tasks completed successfully. Files stored in'"${RESULTDIR}"'"
