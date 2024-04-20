# Diachronic Testing of Causal Language Models

This repository is a fork from [Prompting is not a substitute for probability measurements in large language models](https://github.com/jennhu/metalinguistic-prompting) by Hu & Roger (2023).

Please refer to the author's original implementation for details.

This **work in progress** fork adds backends for `Pythia` and `OLMo` models via the `transformers` library. It is primarily meant for prototyping evaluations using the existing proprocessed datasets from Hu & Levy.

## Additional Models

### EleutherAI/Pythia

The **Pythia** models were trained by [EleutherAI](https://www.eleuther.ai/) in different sizes with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

- [Technical Report on arXiv](https://arxiv.org/abs/2304.01373)
- [Pythia Scaling Suite on HuggingFace](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
- [Offical Github repository](https://github.com/EleutherAI/pythia)

### AI2/OLMo

The **OLMo** models were released by [AI2](https://allenai.org/) in different sizes with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

- [Technical Report on arXiv](https://arxiv.org/abs/2402.00838)
- [OLMo Suite on HuggingFace](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- [Offical Github repository](https://github.com/allenai/OLMo)


## Setup

- requires a GPU with `cuda >= 12.1` support (you can theoretically run smaller models on CPU, but not recommended)

```shell
conda env create -f environment.yml
conda activate meta-w
```

## Evaluation materials

From the original authors:
>Evaluation datasets can be found in the [`datasets`](datasets) folder.
>Please refer to the README in that folder for more details on how the stimuli were assembled and formatted.

## Evaluation scripts

The [scripts](scripts) folder contains scripts for running the experiments. Results of experiments 1 and 2 can be visualized with `new_analysis.ipynb`, but visualization of experiment 3 is currently not supported for the new models *(maybe tbd)*.

### NEW: HuggingFace / Pythia

For example, to evaluate `pythia-70m-deduped` on the *dtfit* dataset for word comparison of Experiment 2, run the following command from the root of this directory:

```shell
bash scripts/run_exp2_pythia.sh blimp EleutherAI/pythia-70m-deduped step143000 pythia-70m-deduped
```
*Note*: `step143000` corresponds to the `main` branch / final checkpoint. See the [HuggingFace Suite](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) for details on how to access different checkpoints.

### NEW: HuggingFace / OLMo

tbd

### HuggingFace / FLAN-T5

The original *HuggingFace* scripts (`*_hf.sh`) utilize the `FLAN-T5` models in 3 different sizes (small, large, XL).

For example, to evaluate `flan-t5-small` on the *SyntaxGym* dataset of Experiment 3b, run the following command from the root of this directory:

```shell
bash scripts/run_exp3b_hf.sh syntaxgym google/flan-t5-small flan-t5-small
```

### OpenAI

The original *OpenAI* implementation (`*_openai.sh`) used 3 different models (`text-curie-001`/GPT-3, `text-davinci-002`/GTP-3.5-SFT, `text-davinci-003`/GTP-3.5-SFT+RLHF) all of which are deprecated by now. 

There are still 2 base models available via *OpenAI*'s API (`babbage-002`/replacement for GPT-3 `ada` and `babbage` base models and `davinci-002`/replacement for GPT-3 `curie` and `davinci` base models), however, the scripts need to be updated first (not done yet). You also need to provide an API key. See the original repository for details.

For more details on the base models still available read the [official documentation](https://platform.openai.com/docs/models/gpt-base).


## ToDo

- currently each eval initializes the model repeatedly, resulting in redundant initializations running several evals at once (e.g. when using the recommended bash scripts) leading to severe slowdown
- no batching: only single instances are passed to the model, again severe slowdown for larger models
- analysis/evaluation with original Jupyter Notebook is kind of broken with new models