from transformers import (
    AutoModelForCausalLM,
    GPTNeoXForCausalLM, 
    AutoTokenizer, 
    T5Tokenizer, 
    T5ForConditionalGeneration,
    BitsAndBytesConfig
)
import torch

torch.set_grad_enabled(False)


# Helper function for loading Huggingface models and tokenizers.
def load_mt(model_name="google/flan-t5-small", revision=None,
            quantization=None, device="cpu", **kwargs):
    # For Pythia and OLMo
    if quantization=="4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
            )
    elif quantization=="8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
            )
    if "flan-t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            **kwargs
            ).to(device)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer ({model_name})")
    elif "pythia" in model_name:
        cache_placeholder = model_name.split('/')[1]
        if quantization in ["4bit", "8bit"]:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=f"./{cache_placeholder}/{revision}",
                device_map="auto",
                quantization_config=quantization_config,
                **kwargs
                )
        else:
            # Full precision
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=f"./{cache_placeholder}/{revision}",
                **kwargs
                ).to(device)
        print(f"Successfully loaded model ({model_name}/{revision})")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=f"./{cache_placeholder}/{revision}",
            )
        print(f"Successfully loaded tokenizer ({model_name}/{revision})")
    elif "allenai" in model_name:
        if quantization in ["4bit", "8bit"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config
            )
        else:
            # Full precision  
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                **kwargs
            ).to(device)
        print(f"Successfully loaded model ({model_name}/{revision})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer ({model_name}/{revision})")
    else:
        print("WARNING: code has only been tested for Flan-T5, Pythia, and OLMo Huggingface models")
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer ({model_name})")
        
    return model, tokenizer

# Wrapper function for creating prompts.
def make_prompt(prefix, continuation, eval_type="direct", task="word_pred", options=None):
    full_sentence = prefix + " " + continuation
    
    # For "direct" evaluation, just use the full sentence.
    if eval_type == "direct":
        return full_sentence
    
    # Otherwise, the prompt structure will depend on the task/experiment.
    if task == "word_pred":
        if eval_type == "metaQuestionSimple":
            prompt = f"What word is most likely to come next in the following sentence?\n\n{full_sentence}"
        elif eval_type == "metaInstruct":
            prompt = f"You are a helpful writing assistant. Tell me what word is most likely to come next in the following sentence:\n\n{full_sentence}"
        elif eval_type == "metaQuestionComplex":
            prompt = f"Here is the beginning of an English sentence: {prefix}... What is the best next word?\n\nAnswer: {continuation}"
        else:
            raise NotImplementedError
            
    elif task == "word_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        else:
            assert len(options) == 2
            option_str = f"{options[0]}, or {options[1]}"
            if eval_type == "metaQuestionSimple":
                prompt = f"What word is most likely to come next in the following sentence ({option_str}?)?\n\n{full_sentence}"
            elif eval_type == "metaInstruct":
                prompt = f"You are a helpful writing assistant. Tell me what word is more likely to come next in the following sentence ({option_str}?):\n\n{full_sentence}"
            elif eval_type == "metaQuestionComplex":
                prompt = f"Here is the beginning of an English sentence: {prefix}... What word is more likely to come next: {option_str}?\n\nAnswer: {continuation}"
            else:
                raise NotImplementedError
                
    elif task == "sentence_judge":
        # Call a separate function for Experiment 3a (sentence judgment in isolation)
        prompt = _make_prompt_yesno(prefix, continuation, eval_type)
                
    elif task == "sentence_comparison":
        if options is None:
            raise ValueError("`options` cannot be None for metalinguistic prompts")
        # Call a separate function for Experiment 3b (sentence comparison)
        prompt = _make_prompt_forcedchoice(continuation, eval_type, options)
    
    else:
        raise ValueError(f"Unknown task specification! '{task}'")
    
    return prompt

def _make_prompt_forcedchoice(continuation, eval_type, options):
    sentence1, sentence2 = options
    option_str = f"1) {sentence1} 2) {sentence2}"
    response_info = "Respond with either 1 or 2 as your answer."
    if eval_type == "metaQuestionSimple":
        prompt = f"Which sentence is a better English sentence? {option_str} {response_info}\n\n{continuation}"
    elif eval_type == "metaInstruct":
        prompt = f"You are a helpful writing assistant. Tell me which sentence is a better English sentence.\n\n{option_str} {response_info}\n\n{continuation}"
    elif eval_type == "metaQuestionComplex":
        prompt = f"Here are two English sentences:\n\n{option_str} Which sentence is a better English sentence? {response_info}\n\nAnswer: {continuation}"
    else:
        raise NotImplementedError
    return prompt

def _make_prompt_yesno(sentence, continuation, eval_type):
    response_info = "Respond with either Yes or No as your answer."
    if eval_type == "metaQuestionSimple":
        prompt = f"Is the following sentence a good sentence of English? {sentence} {response_info}\n\n{continuation}"
    elif eval_type == "metaInstruct":
        prompt = f"You are a helpful writing assistant. Tell me if the following sentence is a good sentence of English.\n\n{sentence} {response_info}\n\n{continuation}"
    elif eval_type == "metaQuestionComplex":
        prompt = f"Here is a sentence:\n\n{sentence} Is the sentence a good sentence of English? {response_info}\n\nAnswer: {continuation}"
    else:
        raise NotImplementedError
    return prompt