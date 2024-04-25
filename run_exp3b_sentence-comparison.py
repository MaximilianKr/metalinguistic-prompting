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
        good_sentence = row.good_sentence
        bad_sentence = row.bad_sentence
        
        if eval_type == "direct":
            # Get standard full-sentence probabilities.
            logprob_of_good_sentence = model.get_full_sentence_logprob(
                good_sentence
            )
            logprob_of_bad_sentence = model.get_full_sentence_logprob(
                bad_sentence
            )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "logprob_of_good_sentence": logprob_of_good_sentence,
                "logprob_of_bad_sentence": logprob_of_bad_sentence
            }
        
        else:
            # Present a particular order of the answer options.
            if option_order == "goodFirst":
                options = [good_sentence, bad_sentence]
            else:
                options = [bad_sentence, good_sentence]
                
            # Create "continuations". We're essentially asking the models
            # a multiple choice question.
            good_continuation = "1" if option_order == "goodFirst" else "2"
            bad_continuation = "2" if option_order == "goodFirst" else "1"
                
            # Create prompt and get outputs.
            good_prompt, logprob_of_good_continuation, logprobs_good = \
                model.get_logprob_of_continuation(
                    "", # no "prefix"
                    good_continuation,
                    task=task,
                    options=options,
                    return_dist=True
                )
            bad_prompt, logprob_of_bad_continuation, logprobs_bad = \
                model.get_logprob_of_continuation( 
                    "", # no "prefix"
                    bad_continuation, 
                    task=task,
                    options=options,
                    return_dist=True
                )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_prompt": good_prompt,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "good_continuation": good_continuation,
                "bad_continuation": bad_continuation,
                "logprob_of_good_continuation": logprob_of_good_continuation,
                "logprob_of_bad_continuation": logprob_of_bad_continuation
            }


         # Record results for this item.
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
    
    task = "sentence_comparison"

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
