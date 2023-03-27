import torch
import timeit
import logging
from transformers import StoppingCriteria


def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid int {input_} set to default: {default}')
        return default


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self._cache_str = ''

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._cache_str += self.tokenizer.decode(input_ids[0, -1])
        for stop_words in self.stop_words:
            if stop_words in self._cache_str:
                return True
        return False


def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid float {input_} set to default: {default}')
        return default

def post_processing_text(output_text, stop_tokens, denylist = []):
    logging.debug(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    logging.debug(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    logging.debug(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    logging.debug(f"<post_processing_text>2 end_pos: {end_pos}.")
    logging.debug(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    logging.debug(f"<post_processing_text> input: {output_text}")
    logging.debug(f"<post_processing_text> output: {post_processed_text}")
    start = timeit.default_timer()
    for word in denylist:
        if post_processed_text.find(word) != -1:
            print(f"<post_processing_text> post_processed_text: {post_processed_text}")
            print(f"<post_processing_text> denylist word {word} found, set to empty.")
            post_processed_text = "Sorry, I'm not sure how to answer that question."
            break
    stop = timeit.default_timer()
    print(f"<post_processing_text> time: {stop - start}")
    return post_processed_text


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    results = []
    batch_size = scores[0].shape[0]
    print(f"<convert_hf_score_to_logprobs>: batch size: {batch_size}")

    for i in range(batch_size):
        logprobs = []
        for current_step_score in scores[i:i+1]:
            value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score.float()), dim=-1), k)
            current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
            logprobs.append(current_logprob)
        results.append(logprobs)
    return results


def torch_dtype_from_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "int8":
        return "llm.int8"
    else:
        return torch.float32
