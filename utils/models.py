import openai
import torch
import numpy as np

from minicons import scorer
from .llm import load_mt, make_prompt


# Base class for large language model.
class LLM(object):
    def __init__(self, model, seed, device="cpu"):
        self.eval_type = None
        self.model = model
        self.seed = seed
        self.device = device

    def get_logprob_of_continuation(self, _):
        raise NotImplementedError
        
    def get_full_sentence_logprob(self, _):
        raise NotImplementedError
    

class OpenAI_LLM(LLM):
    # Helper function for obtaining token-by-token log probabilities.
    def _get_logprobs(self, prompt, **kwargs):
        completion = openai.Completion.create(
            prompt=prompt,
            model=self.model,
            logprobs=5, # 5 is the maximum (see OpenAI API docs)
            max_tokens=0,
            echo=True,
            **kwargs
        ).choices[0]
        logprobs = completion.logprobs
        return logprobs
    
    def get_full_sentence_logprob(self, sentence, **kwargs):
        logprobs = self._get_logprobs(sentence)
        token_logprobs, top_logprobs = \
            logprobs["token_logprobs"], logprobs["top_logprobs"]

        # Sum up logprobs for each token in the sentence to get the full logprob.
        relevant_token_logprobs = token_logprobs[1:] # the first entry is None; no prob for first token
        total_logprob = sum(relevant_token_logprobs)
        
        return total_logprob
    
    def get_logprob_of_continuation(self,
                                    prefix, 
                                    continuation, 
                                    task="word_pred",
                                    options=None, 
                                    return_dist=True,
                                    **kwargs):
        # Construct prompt and get logprobs.
        prompt = make_prompt(
            prefix, 
            continuation,
            eval_type=self.eval_type,
            task=task,
            options=options
        )
        logprobs = self._get_logprobs(prompt, **kwargs)
        tokens, token_logprobs, top_logprobs = \
            logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"]

        # Identify indices from `tokens` that correspond to the relevant
        # continuation (final word). This could be split into multiple tokens.
        n_tokens = len(tokens)
        full_continuation_str = " " + continuation
        if task == "sentence_comparison":
            # The number tokens sometimes have preceding space, sometimes not.
            end_strs = [full_continuation_str, continuation] 
        else:
            end_strs = full_continuation_str
        inds = []
        cur_word = ""
        for tok_idx in range(n_tokens-1, -1, -1):
            # Go backwards through the list of tokens.
            cur_tok = tokens[tok_idx]
            cur_word = cur_tok + cur_word
            if token_logprobs[tok_idx] is None:
                break
            else:
                inds = [tok_idx] + inds
                if cur_word in end_strs:
                    break
        
        # Obtain logprob of gold (ground-truth) word by summing logprobs
        # of all sub-word tokens, as measured by `inds`.
        logprob_of_continuation = sum([token_logprobs[i] for i in inds])
        
        # Optionally return top 5 logprobs (maximum allowed by OpenAI).
        if return_dist:
            # Get top 5 logprobs for each relevant subword token.
            top_logprobs = [top_logprobs[i] for i in inds]
            return prompt, logprob_of_continuation, top_logprobs
        else:
            return prompt, logprob_of_continuation
        
class Causal_LLM(LLM):
    def __init__(self, model, revision, quantization, seed, device="cpu"):
        super().__init__(model, seed, device)
        self.model = scorer.IncrementalLMScorer(
            model=model, 
            device=device, 
            revision=revision
            )
        self.eval_type = ""

    def _get_logprobs(self, prompt):
        pass

    def get_full_sentence_logprob(self, sentence):
        pass

    def get_logprob_of_continuation(self,
                                    prefix, 
                                    continuation, 
                                    task="word_pred",
                                    options=None, 
                                    return_dist=False,
                                    ):
        # Construct prompt and get logprobs
        prompt = make_prompt(
            prefix, 
            continuation,
            eval_type=self.eval_type,
            task=task,
            options=options
        )

        # Sequence Log-probability
        # reduction = lambda x: x.sum(0).item()
        # see https://github.com/kanishkamisra/minicons
        logprob_of_continuation = self.model.sequence_score(
            prompt, 
            reduction = lambda x: x.sum(0).item()
            )

        if return_dist:
            full_vocab_logprobs = []  # TODO: maybe fixme
            return prompt, logprob_of_continuation[0], full_vocab_logprobs
        else:
            return prompt, logprob_of_continuation[0]


class Pythia_LLM(LLM):
    # TODO: add batch processing?
    def __init__(self, model, revision, quantization, seed, device="cpu"):
        super().__init__(model, seed, device)
        self._model, self._tokenizer = load_mt(model, revision, quantization, device)
        self.eval_type = ""
        self._model.eval()

    def _get_logprobs(self, prompt): #**kwargs):
        # TODO: is this logic of getting logprobs correct?
        # adapted from https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
        # Tokenize input prompt
        encoding = self._tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)

        # Process input through model
        outputs = self._model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        probs = probs[0, :-1, :]
        sliced_input_ids = input_ids[0, 1:]
        gen_probs = torch.gather(probs, 1, sliced_input_ids[:, None]).squeeze(-1)

        # Restore first token
        first_token_id = input_ids[0, 0]
        first_token = self._tokenizer.decode(first_token_id)

        # Collect tokens and their log probabilities into structured dictionary
        tokens = []
        tokens.append(first_token)
        token_logprobs = [None]  # First token will have no logprob associated
        for token, p in zip(sliced_input_ids, gen_probs):
            if token not in self._tokenizer.all_special_ids:
                decoded_token = self._tokenizer.decode([token])
                tokens.append(decoded_token)
                token_logprobs.append(p.item())

        return {"tokens": tokens, "token_logprobs": token_logprobs}

    def get_full_sentence_logprob(self, sentence):
        logprobs = self._get_logprobs(sentence)
        token_logprobs = logprobs["token_logprobs"]

        # Sum up logprobs for each token in the sentence to get the full logprob.
        relevant_token_logprobs = token_logprobs[1:] # the first entry is None; no prob for first token
        total_logprob = sum(relevant_token_logprobs)
        
        return total_logprob

    def get_logprob_of_continuation(self,
                                    prefix, 
                                    continuation, 
                                    task="word_pred",
                                    options=None, 
                                    return_dist=False,
                                    ):
        # Construct prompt and get logprobs
        prompt = make_prompt(
            prefix, 
            continuation,
            eval_type=self.eval_type,
            task=task,
            options=options
        )
        logprobs = self._get_logprobs(prompt)
        tokens, token_logprobs = logprobs["tokens"], logprobs["token_logprobs"]

        # Identify indices from `tokens` that correspond to the relevant
        # continuation (final word). This could be split into multiple tokens.
        n_tokens = len(tokens)
        full_continuation_str = " " + continuation
        if task == "sentence_comparison":
            # The number tokens sometimes have preceding space, sometimes not.
            end_strs = [full_continuation_str, continuation] 
        else:
            end_strs = full_continuation_str
        inds = []
        cur_word = ""
        for tok_idx in range(n_tokens-1, -1, -1):
            # Go backwards through the list of tokens.
            cur_tok = tokens[tok_idx]
            cur_word = cur_tok + cur_word
            if token_logprobs[tok_idx] is None:
                break
            else:
                inds = [tok_idx] + inds
                if cur_word in end_strs:
                    break

        # Obtain logprob of gold (ground-truth) word by summing logprobs
        # of all sub-word tokens, as measured by `inds`.
        logprob_of_continuation = sum([token_logprobs[i] for i in inds])
        
        if return_dist:
            full_vocab_logprobs = []  # TODO: maybe fixme
            return prompt, logprob_of_continuation, full_vocab_logprobs
        else:
            return prompt, logprob_of_continuation


class OLMo_LLM(Pythia_LLM):
    # Same functionality as Pythia models, but could be extended later
    pass


class T5_LLM(LLM):
    def __init__(self, model, seed, device="cpu", ignore_special_logprobs=True):
        super().__init__(model, seed, device=device)
        self._model, self._tokenizer = load_mt(self.model, device=self.device)
        self._model.eval()
        
        if ignore_special_logprobs:
            self.tokens_to_ignore = ["<extra_id_0>", "<extra_id_1>", "</s>"]
            self.ids_to_ignore = self._tokenizer.convert_tokens_to_ids(
                self.tokens_to_ignore
            )
        
    def get_full_sentence_logprob(self, sentence, **kwargs):
        # Chop off period and split into words based on whitespace.
        # NOTE: this works for the simple sentences in our stimuli, but could be changed for more naturalistic data.
        if sentence.endswith("."):
            sentence = sentence[:-1]
        words = sentence.split(" ")
        
        # Pseudolikelihood method: "mask out" and predict each token, one by one.
        total_logprob = 0
        for i, w in enumerate(words):
            # Create input and output strings using masked T5 format.
            inpt_str = " ".join(words[:i]) + " <extra_id_0> " + " ".join(words[i+1:])
            inpt_str = inpt_str.strip()
            if i == 0:
                output_str = f"{w} <extra_id_0>"
            elif i == len(words) - 1:
                output_str = f"<extra_id_0> {w}"
            else:
                output_str = f"<extra_id_0> {w} <extra_id_1>"
            
            # Tokenize the inputs and labels.
            input_ids = self._tokenizer(inpt_str, return_tensors="pt").input_ids.to(self.device)
            labels = self._tokenizer(output_str, return_tensors="pt").input_ids.to(self.device)

            # Model forward.
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, labels=labels, **kwargs)

            # Turn logits into log probabilities.
            logits = outputs.logits
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Subset the labels and logprobs we care about,
            # i.e. the non-"special" tokens (e.g., "<extra_id_0>").
            mask = torch.BoolTensor([tok_id not in self.ids_to_ignore for tok_id in labels[0]])
            relevant_labels = labels[0][mask]
            relevant_logprobs = logprobs[0][mask]
            
            # Index into logprob tensor using the relevant token IDs.
            logprobs_to_sum = [
                relevant_logprobs[i][tok_id] 
                for i, tok_id in enumerate(relevant_labels)
            ]
            total_logprob += sum(logprobs_to_sum).item()
            
        return total_logprob
        
    def get_logprob_of_continuation(self,
                                    prefix, 
                                    continuation, 
                                    task="word_pred",
                                    options=None, 
                                    return_dist=True,
                                    **kwargs):
        # Construct prompt and get logprobs.
        full_prompt = make_prompt(
            prefix, 
            continuation,
            eval_type=self.eval_type,
            task=task,
            options=options
        )
        inpt_str = make_prompt(
            prefix, 
            "<extra_id_0>", 
            eval_type=self.eval_type,
            task=task,
            options=options
        )
        if full_prompt.endswith(continuation):
            output_str = f"<extra_id_0> {continuation}"
        else:
            output_str = f"<extra_id_0> {continuation} <extra_id_1>"
        
        # Tokenize the inputs and labels.
        input_ids = self._tokenizer(inpt_str, return_tensors="pt").input_ids.to(self.device)
        labels = self._tokenizer(output_str, return_tensors="pt").input_ids.to(self.device)
        
        # Model forward.
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, labels=labels, **kwargs)

        # Turn logits into log probabilities.
        logits = outputs.logits
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # IGNORE FIRST TOKEN: this corresponds to <extra_id_0>
        relevant_labels = labels[0][1:]
        relevant_logprobs = logprobs[0][1:]
        # Also ignore <extra_id_1> at the end, if there's extra content
        if not full_prompt.endswith(continuation):
            relevant_labels = relevant_labels[:-1]
            relevant_logprobs = relevant_logprobs[:-1]

        # Index into logprob tensor using the relevant token IDs.
        logprobs_to_sum = [
            relevant_logprobs[i][tok_id] 
            for i, tok_id in enumerate(relevant_labels)
        ]
        logprob_of_continuation = sum(logprobs_to_sum).item()

        # Optionally return full distribution. Only keep the distribution 
        # corresponding to the first subword token.
        if return_dist:
            full_vocab_logprobs = relevant_logprobs[0]
            return full_prompt, logprob_of_continuation, full_vocab_logprobs
        else:
            return full_prompt, logprob_of_continuation
        
    def save_dist_as_numpy(self, dist, path):
        dist = dist.detach().numpy()
        np.save(path, dist)
