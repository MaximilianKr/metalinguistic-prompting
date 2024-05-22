import json
from time import gmtime, strftime
import torch

from . import models  # openai_api

torch.set_grad_enabled(False)
torch.random.manual_seed(42)


# Extremely basic helper functions.
def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def dict2json(d, out_file):
    with open(out_file, "w") as fp:
        json.dump(d, fp, indent=2)

def json2dict(in_file):
    with open(in_file, "r") as fp:
        d = json.load(fp)
    return d

def initialize_model(model_name, revision, quantization, seed):
    # Set device to GPU if cuda is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Set device to CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA unvailable); adjust your expectations")
        
    # TODO: add backends: 'togethercomputer/RedPajama-INCITE-7B-Base'
    #                     'TinyLlama/TinyLlama-1.1B'
    #                     'Zyphra/Zamba-7b'

    # TODO: fix OpenAI
    # if model_name == "openai":
    #     # Secret file with API key (DO NOT commit this)
    #     openai_api.set_key_from_file(args.key)
    #     model = models.OpenAI_LLM(args.eval_type, args.model, args.seed)
    if "flan-t5" in model_name:
        model = models.T5_LLM(model=model_name, seed=seed, device=device)
    elif "pythia" in model_name:
        model = models.Pythia_LLM(model=model_name, revision=revision, quantization=quantization, seed=seed, device=device)
    elif "allenai" in model_name:
        model = models.OLMo_LLM(model=model_name, revision=revision, quantization=quantization, seed=seed, device=device)
    else:
        raise ValueError(
            f"Model not supported! (Your model: {model_name})"
        )
    return model
