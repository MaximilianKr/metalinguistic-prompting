#!/bin/bash

# Required, "syntaxgym" or "blimp"
CORPUS=$1

# Required, "EleutherAI/pythia-70m-deduped", "allenai/OLMo-1B-hf", "google/flan-t5-small"
MODEL=$2

# Optional, default "main" / see intermediate checkpoints for models
REVISION=${3:-"main"}

# Optional, default "full" precision, possible "4bit" or "8bit"
QUANTIZATION=${4:-"full"}

SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
RESULTDIR="results/exp3a_sentence-judgment"
DATAFILE="datasets/exp3/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

if [ "$QUANTIZATION" != "full" ]; then
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}"
else
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${REVISION}"
fi

python run_exp3a_sentence-judgment.py $MODEL $REVISION $QUANTIZATION $DATAFILE $OUTFILE

echo "All tasks completed successfully. Files stored in'"${RESULTDIR}"'"


# # Helper function
# run_experiment () {
#     # Capture relevant variables
#     local EVAL_TYPE=$1

#     # Define variable-dependent file/folder names
#     OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}.json"
    
#     # By default, we won't save the full vocab distributions.
#     # Uncomment the two lines below if you'd like to.
    
#     # DISTFOLDER="${RESULTDIR}/dists/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}"
#     # mkdir -p $DISTFOLDER

#     # Run the evaluation script
#     echo "Running Experiment 3a (sentence judgment): model = ${MODEL}; eval_type = ${EVAL_TYPE}" >&2
#     python run_exp3a_sentence-judgment.py \
#         --model $MODEL \
#         --model_type "hf" \
#         --eval_type ${EVAL_TYPE} \
#         --data_file $DATAFILE --out_file ${OUTFILE}
#         # Uncomment the line below to save full distributions.
#         # --dist_folder $DISTFOLDER
# }

# NOTE: "direct" model is THE SAME across 3a and 3b
# FIRST, run the "direct" model.
run_experiment "direct"

# # NEXT, run the other models.
for EVAL_TYPE in "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    run_experiment "${EVAL_TYPE}"
done
