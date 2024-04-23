# Diachronic Testing of Causal Language Models

This repository is a fork from [Prompting is not a substitute for probability measurements in large language models](https://github.com/jennhu/metalinguistic-prompting) by Hu & Levy (2023).

Please refer to the author's original implementation for details.

This **work in progress** fork adds backends for `Pythia` and `OLMo` (with option for running quantized) models via the `transformers` library. It is primarily meant for prototyping evaluations using the existing preprocessed datasets from Hu & Levy.

## Overview

- [Additional Models](#additional-models)
  - [EleutherAI/Pythia](#eleutherai-pythia)
  - [AI2/OLMo](#ai2-olmo)
- [Setup](#setup)
  - [venv](#venv)
  - [conda](#conda)
- [Evaluation materials](#evaluation-materials)
- [Evaluation scripts](#evaluation-scripts)
  - [NEW: HuggingFace / Pythia](#new-huggingface--pythia)
  - [NEW: HuggingFace / OLMo](#new-huggingface--olmo)
    - [Full Precision](#full-precision)
    - [Quantized Model](#quantized-model-either-4bit-or-8bit)
  - [OLD: HuggingFace / FLAN-T5](#old-huggingface--flan-t5)
  - [OLD: OpenAI](#old-openai)
- [ToDo](#todo)
- [Author](#author)

## Additional Models

### EleutherAI-Pythia

The **Pythia** models were trained by [EleutherAI](https://www.eleuther.ai/) in different sizes with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

- [Technical Report on arXiv](https://arxiv.org/abs/2304.01373)
- [Pythia Scaling Suite on HuggingFace](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
- [Offical Github repository](https://github.com/EleutherAI/pythia)

### AI2-OLMo

The **OLMo** models were released by [AI2](https://allenai.org/) in different sizes with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

- [Technical Report on arXiv](https://arxiv.org/abs/2402.00838)
- [OLMo Suite on HuggingFace](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- [Offical Github repository](https://github.com/allenai/OLMo)

## Setup

- requires a GPU with `cuda >= 12.1` support (you can theoretically run smaller models on CPU, but not recommended)

### venv

- use [uv package manager]((https://github.com/astral-sh/uv)) for a fast setup
```shell
uv venv
```
```shell
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```
```shell
uv pip install -r requirements.txt
```

### conda

```shell
conda env create -f environment.yml
conda activate metalinguistic-prompting
```

## Evaluation materials

From the original authors:
>Evaluation datasets can be found in the [`datasets`](datasets) folder.
>Please refer to the README in that folder for more details on how the stimuli were assembled and formatted.

## Evaluation scripts

The [scripts](scripts) folder contains scripts for running the experiments. Results of Experiments 1, 2, and 3b can be visualized with `new_analysis.ipynb`, but visualization of Experiment 3a (isolated instances) is currently not supported for the new models *(maybe tbd)*.

### NEW: HuggingFace / Pythia

```shell
# Template 
bash scripts/<experiment_script>.sh <corpus> EleutherAI/<pythia-model> <revision> <save_name>
```

For example, to evaluate `pythia-70m-deduped` on the *dtfit* dataset for the word comparison of Experiment 2, run the following command from the root of this directory:

```shell
bash scripts/run_exp2_pythia.sh blimp EleutherAI/pythia-70m-deduped main pythia-70m-deduped
```

*Note*: revision `main` corresponds to the main branch / final checkpoint. Check one of the [HuggingFace Pythia Model Cards](https://huggingface.co/EleutherAI/pythia-70m-deduped) for details on how to access different (earlier) checkpoints.

*Note 2*: In theory, quantization also works for Pythia models and the code is implemented (see equivalent [instructions](#quantized-model-either-4bit-or-8bit) below for OLMo models). However, in its current form, loading Pythia checkpoint shards is very slow (see [ToDo](#todo)).

### NEW: HuggingFace / OLMo

```shell
# Template
bash scripts/<experiment_script>.sh <corpus> allenai/<OLMo-model> <revision> <save_name> <optional: quantization>
```

#### Full Precision

For example, to evaluate `OLMo-1B-hf` (with full precision) on the *news* dataset for the word prediction of Experiment 1, run the following command from the root of this directory:

```shell
bash scripts/run_exp1_olmo.sh news allenai/OLMo-1B-hf main OLMo-1B-hf
```

*Note*: revision `main` corresponds to the main branch / final checkpoint. Check one of the [HuggingFace OLMo Model Cards](https://huggingface.co/allenai/OLMo-1.7-7B-hf) for details on how to access different (earlier) checkpoints.

#### Quantized Model (either `4bit` or `8bit`)

For example, to evaluate `OLMo-7B-hf` with **4bit precision** on the *p18* dataset for the word prediction of Experiment 1, run the following command from the root of this directory:

```shell
bash scripts/run_exp1_olmo.sh p18 allenai/OLMo-7B-hf main OLMo-7B-hf 4bit
```

### OLD: HuggingFace / FLAN-T5

The original *HuggingFace* scripts (`*_hf.sh`) utilize the `FLAN-T5` models in 3 different sizes (small, large, XL).

For example, to evaluate `flan-t5-small` on the *SyntaxGym* dataset of Experiment 3b, run the following command from the root of this directory:

```shell
bash scripts/run_exp3b_hf.sh syntaxgym google/flan-t5-small flan-t5-small
```

### OLD: OpenAI

The original *OpenAI* implementation (`*_openai.sh`) used 3 different models (`text-curie-001`/GPT-3, `text-davinci-002`/GTP-3.5-SFT, `text-davinci-003`/GTP-3.5-SFT+RLHF) all of which are deprecated by now.

There are still 2 base models available via *OpenAI*'s API (`babbage-002`/replacement for GPT-3 `ada` and `babbage` base models and `davinci-002`/replacement for GPT-3 `curie` and `davinci` base models), however, the scripts need to be updated first (not done yet). You also need to provide an API key. See the original repository for details.

For more details on the base models still available read the [official documentation](https://platform.openai.com/docs/models/gpt-base).

## ToDo

- **Repeated model initializations**: currently each eval initializes the model repeatedly, resulting in redundant initializations when running several evals at once (e.g. when using the recommended bash scripts) leading to severe slowdown
- **Pythia quantization very slow**: while the scripts run with quantized Pythia models, loading checkpoint shards takes too long, there must be something wrong with loading the quantized models for Pythia (works for OLMo though)
- **Missing batching**: only single instances are passed to the model, possible improvements achievable (especially for larger models)
- **Analysis/Evaluation broken**: original Jupyter Notebook is kind of broken with new models, evaluation for Experiment 3a (isolated) does not work
- **Clean up code**: there is some boilerplate / redundant code + script files in there (and I added even more)

## Author

Please refer to the authors of the [original repository](https://github.com/jennhu/metalinguistic-prompting)

For the fork:

- Maximilian Krupop

[Back to Top](#diachronic-testing-of-causal-language-models)
