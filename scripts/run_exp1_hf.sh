#!/bin/bash

# Required, "p18" or "news"
CORPUS=$1

# Required, "EleutherAI/pythia-70m-deduped", "allenai/OLMo-1B-hf", "google/flan-t5-small"
MODEL=$2

# Optional, default "main" / see intermediate checkpoints for models
REVISION=${3:-"main"}

# Optional, default "full" precision, possible "4bit" or "8bit"
QUANTIZATION=${4:-"full"}

SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
RESULTDIR="results/exp1_word-prediction"
DATAFILE="datasets/exp1/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

if [ "$QUANTIZATION" != "full" ]; then
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}"
else
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${REVISION}"
fi

python run_exp1_word-prediction.py $MODEL $REVISION $QUANTIZATION $DATAFILE $OUTFILE

echo "All tasks completed successfully. Files stored in'"${RESULTDIR}"'"

# for EVAL_TYPE in "direct" "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
#     OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}.json"
    
#     # By default, we won't save the full vocab distributions.
#     # Uncomment the two lines below if you'd like to.
    
#     # DISTFOLDER="${RESULTDIR}/dists/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}"
#     # mkdir -p $DISTFOLDER

#     echo "Running Experiment 1 (word prediction): model = ${MODEL}; eval_type = ${EVAL_TYPE}"
#     python run_exp1_word-prediction.py \
#         --model $MODEL \
#         --model_type "hf" \
#         --eval_type ${EVAL_TYPE} \
#         --data_file $DATAFILE --out_file ${OUTFILE}
#         # Uncomment the line below to save full distributions.
#         # --dist_folder $DISTFOLDER
# done