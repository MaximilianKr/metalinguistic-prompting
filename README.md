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
  - [Pythia / OLMo](#pythia--olmo-models)
  - [FLAN-T5](#old-flan-t5)
  - [OpenAI](#old-openai)
- [ToDo](#todo)
- [Author](#author)

## Additional Models

Both models were released in different parameter sizes and with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

### EleutherAI-Pythia

- [arXiv Technical Report](https://arxiv.org/abs/2304.01373)
- [HuggingFace Pythia Scaling Suite](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
- [Github pythia](https://github.com/EleutherAI/pythia)
- [EleutherAI](https://www.eleuther.ai/)

### AI2-OLMo

- [arXiv Technical Report](https://arxiv.org/abs/2402.00838)
- [HuggingFace OLMo Suite](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- [Github OLMo](https://github.com/allenai/OLMo)
- [AI2](https://allenai.org/)

## Setup

- requires GPU with `cuda >= 12.1` support (smaller models can run on CPU, but not recommended)

### venv

- use [uv package manager](https://github.com/astral-sh/uv) for a fast setup

```shell
uv venv
```

```shell
# macOS / Linux
source .venv/bin/activate
```

```shell
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

### Experiments

- Experiment 1

  ```shell
  # Template
  # bash scripts/run_exp1_hf.sh {p18, news} {huggingface/model} {revision}  # optional quantization: {4bit, 8bit}
  
  # Example
  bash scripts/run_exp1_hf.sh news EleutherAI/pythia-70m-deduped main
  ```

- Experiment 2

  ```shell
  # Template
  # bash scripts/run_exp2_hf.sh {huggingface/model} {revision}  # optional quantization: {4bit, 8bit}
  
  # Example
  bash scripts/run_exp2_hf.sh allenai/OLMo-7B-hf main 8bit
  ```

- Experiment 3

  ```shell
  # Template
  # bash scripts/run_exp3{a, b}_hf.sh {syntaxgym, blimp} {huggingface/model} {revision}  # optional quantization: {4bit, 8bit}
  
  # Example
  bash scripts/run_exp3a_hf.sh EleutherAI/pythia-2.8B-deduped step3000 main
  ```

#### Pythia & OLMo models

- `revision`

  - `main` corresponds to the final model checkpoint. Check either [Pythia](https://huggingface.co/EleutherAI/pythia-70m-deduped) or [OLMo](https://huggingface.co/allenai/OLMo-1.7-7B-hf) model cards on Huggingface for details on how to access different (earlier) checkpoints.

- `quantization`
  
  - `8bit` or `4bit`, running with less precision but with less VRAM needed. Loading checkpoint shards can take longer than with pull precision (*quantized OLMo models load fine, Pythia models very slow*)

### Old: FLAN-T5

*probably needs fixing*

The original *HuggingFace* scripts (`*_hf.sh`) utilize the `FLAN-T5` models in 3 different sizes (small, large, XL).

For example, to evaluate `flan-t5-small` on the *SyntaxGym* dataset of Experiment 3b, run the following command from the root of this directory:

```shell
bash scripts/run_exp3b_hf.sh syntaxgym google/flan-t5-small flan-t5-small
```

### Old: OpenAI

*probably needs fixing*

The original *OpenAI* implementation (`*_openai.sh`) used 3 different models (`text-curie-001`/GPT-3, `text-davinci-002`/GTP-3.5-SFT, `text-davinci-003`/GTP-3.5-SFT+RLHF) all of which are deprecated by now.

There are still 2 base models available via *OpenAI*'s API (`babbage-002`/replacement for GPT-3 `ada` and `babbage` base models and `davinci-002`/replacement for GPT-3 `curie` and `davinci` base models), however, the scripts need to be updated first (not done yet). You also need to provide an API key. See the original repository for details.

For more details on the base models still available read the [official documentation](https://platform.openai.com/docs/models/gpt-base).

## ToDo

- [ ] test [minicons](https://github.com/kanishkamisra/minicons) implementation

- [ ] test **instruct-tuned models** for all other prompting techniques than *direct*

- [ ] fix **Pythia quantization slow** loading quantized checkpoint shards for Pythia takes too long (works for OLMo though)

- [ ] add **batching support** only single instances passed to the model, possible improvements achievable (especially for larger models)

- [ ] fix old scripts restore Flan-T5 and OpenAI support

- [ ] fix `analysis.ipynb`` original notebook broken with new models, evaluation for Experiment 3a (isolated) does not work

- [ ] clean up code

## Author

Please refer to the authors of the [original repository](https://github.com/jennhu/metalinguistic-prompting)

For the fork:

- Maximilian Krupop

[Back to Top](#diachronic-testing-of-causal-language-models)
