import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import io


def run_experiment(model, out_file: str, meta_data: dict):
    eval_type = meta_data["eval_type"]
    task = meta_data["task"]
    option_order = meta_data["option_order"]

    model.eval_type = eval_type
    
    # Read corpus data.
    df = pd.read_csv(meta_data["data_file"])

    results = []
    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        # Present a particular order of the answer options.
        if option_order == "goodFirst":
            options = [row.good_continuation, row.bad_continuation]
        else:
            options = [row.bad_continuation, row.good_continuation]

        # Create prompt and get outputs.
        # TODO: fix 'resturn_dist'
        good_prompt, logprob_of_good_continuation, logprobs_good = \
            model.get_logprob_of_continuation(
                row.prefix, 
                row.good_continuation, 
                task=task,
                options=options,
                return_dist=True,
            )
        bad_prompt, logprob_of_bad_continuation, logprobs_bad = \
            model.get_logprob_of_continuation(
                row.prefix, 
                row.bad_continuation, 
                task=task,
                options=options,
                return_dist=True,
            )
        
        # Store results in dictionary.
        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "good_prompt": good_prompt,
            "good_continuation": row.good_continuation,
            "bad_continuation": row.bad_continuation,
            "logprob_of_good_continuation": logprob_of_good_continuation,
            "logprob_of_bad_continuation": logprob_of_bad_continuation
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
    
    task = "word_comparison"

    # Define experiments
    experiments = [
        ("direct", "goodFirst"),
        ("metaQuestionSimple", "goodFirst"),
        ("metaQuestionSimple", "badFirst"),
        ("metaInstruct", "goodFirst"),
        ("metaInstruct", "badFirst"),
        ("metaQuestionComplex", "goodFirst"),
        ("metaQuestionComplex", "badFirst"),
    ]

    # Run experiments
    for eval_type, option_order in experiments:
        meta_data = {
            "model": model_name,
            "revision": revision,
            "quantization": quantization,
            "seed": seed,
            "task": task,
            "eval_type": eval_type,
            "option_order": option_order,
            "data_file": data_file,
            "timestamp": io.timestamp()
        }

        final_out = f"{outfile}_{eval_type}_{option_order}.json"

        print(f"Running '{task}', '{eval_type}', '{option_order}'")
        run_experiment(
            model, final_out, meta_data
            )
        

if __name__ == '__main__':
    main()
