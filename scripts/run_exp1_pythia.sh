#!/bin/bash

CORPUS=$1  # "p18" or "news"
MODEL=$2  # e.g., "EleutherAI/pythia-70m-deduped"
REVISION=$3  # e.g., "step3000", main branch / last checkpoint is "step143000"
SAFEMODEL=$4  # e.g., "pythia-70m-deduped"; this should be safe for file-naming purposes
QUANTIZATION=${5:-"full"}  # "4bit" or "8bit", optional

RESULTDIR="results/exp1_word-prediction"
DATAFILE="datasets/exp1/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

for EVAL_TYPE in "direct" "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${QUANTIZATION}_${REVISION}_${EVAL_TYPE}.json"
    
    # By default, we won't save the full vocab distributions.
    # Uncomment the two lines below if you'd like to.
    
    # DISTFOLDER="${RESULTDIR}/dists/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}"
    # mkdir -p $DISTFOLDER

    echo "Running Experiment 1 (word prediction): model = ${MODEL}_${QUANTIZATION}_${REVISION}; eval_type = ${EVAL_TYPE}"
    python run_exp1_word-prediction.py \
        --model $MODEL \
        --revision $REVISION \
        --quantization $QUANTIZATION \
        --model_type "hf" \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE --out_file ${OUTFILE}
        # Uncomment the line below to save full distributions.
        # --dist_folder $DISTFOLDER
done
