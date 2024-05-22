import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

from utils import io


def run_experiment(model, out_file: str, meta_data: dict):
    eval_type = meta_data["eval_type"]
    task = meta_data["task"]

    model.eval_type = eval_type
    
    # Read corpus data.
    df = pd.read_csv(meta_data["data_file"])
    
    results = []
    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        # Create prompt and get outputs.
        prompt, logprob_of_continuation, logprobs = \
            model.get_logprob_of_continuation(
                row.prefix,
                row.continuation,
                task=task,
                options=None,
                return_dist=True
            )
        
        # Store results in dictionary.
        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "prompt": prompt,
            "gold_continuation": row.continuation,
            "logprob_of_gold_continuation": logprob_of_continuation,
        }

        # Record results for current item.
        results.append(res)
    
    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    io.dict2json(output, out_file)
  


def main():
    if len(sys.argv) < 5:
        print("Usage:\nbash scripts/<experiment_script>.sh <huggingface/model> <optional:revision> <optional: quantization>") 
        sys.exit(1)

    # For reproducability
    seed = np.random.seed(42)

    model_name, revision, quantization, data_file, outfile = sys.argv[1:6]

    model = io.initialize_model(model_name, revision, quantization, seed)
    
    task = "word_pred"

    # Define experiments
    experiments = [
        ("direct"),
        ("metaQuestionSimple"),
        ("metaInstruct"),
        ("metaQuestionComplex"),
    ]

    # Run experiments
    for eval_type in experiments:
        meta_data = {
            "model": model_name,
            "revision": revision,
            "quantization": quantization,
            "seed": seed,
            "task": task,
            "eval_type": eval_type,
            "data_file": data_file,
            "timestamp": io.timestamp()
        }

        final_out = f"{outfile}_{eval_type}.json"

        print(f"Running '{task}', '{eval_type}'")
        run_experiment(
            model, final_out, meta_data
            )
        

if __name__ == '__main__':
    main()